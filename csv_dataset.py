# csv_dataset.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

class NetworkTraffic():
    '''
    Class for Network Traffic Dataset
    '''

    dataset_files = [] # csv dataset file name list
    features = [] # dataset features
    dataset = pd.DataFrame([[]]) # pandas dataframe type dataset

    def __init__(self, csv_files, features):
        '''
        constructor for NetworkTraffic class
        :param csv_files: csv dataset file path list; list type
        :param features: featrues of dataset; list type
        '''
        self.features = features
        self.dataset_files = csv_files

    def count_data(self, col_name, col_values, header=None):
        '''
        count data that is equal to col_values in col_name
        :param col_name: column name that function will count; str type
        :param col_values: column values for count condition; list type
        :param header: when dataset file has header pandas can infer it as column names
        :return: counted data which satisfied each col_value in col_values
        '''
        features = self.features
        if header is not None:
            features = None

        counts = [0 for i in range(len(col_values))]
        for dataset_file in self.dataset_files: # we do not have to load all dataset files to memory
            buf = pd.read_csv(dataset_file, names=features, header=header)
            for i in range(len(col_values)):
                counts[i] += len(buf[buf[col_name] == col_values[i]])
        return counts

    def set_dataset(self, header=None):
        '''
        make pandas dataframe typed dataset specified in dataset_files attribute
        :param header: if there is a header in the dataset file, pandas can infer it as columns;
        None (default) or "infer"
        :return: None
        '''
        data_frames = []
        features = self.features

        if header is not None:
            features = None

        for csv in self.dataset_files:
            data_frames.append(pd.read_csv(csv, names=features, header=header))

        self.dataset = pd.concat(data_frames, ignore_index=True)

    def unsw_dataset_preprocessor(self, to_category, missing_features, drop_cols):
        '''
        preprocessor for UNSW IoT-Botnet Dataset
        :param to_category: feature list to categorize; list type
        :param missing_features: feature list that is missing; list type
        :param drop_cols: feature list to drop; list type
        :return: None
        '''
        def ipv6_mac_to_int(x):
            if ':' in str(x):
                return str(int(x.replace(':', ''), 16))
            else:
                return x

        def hex_to_dec(x):
            if '0x' in str(x):
                return int(str(x), 16)
            else:
                return x

        self.dataset = pd.get_dummies(self.dataset, columns=to_category)
        self.dataset = self.dataset.drop(missing_features, axis=1)
        self.dataset = self.dataset.drop(drop_cols, axis=1)

        self.dataset["sport"] = self.dataset["sport"].fillna(-1)
        self.dataset["sport"] = self.dataset["sport"].apply(hex_to_dec)
        self.dataset["dport"] = self.dataset["dport"].fillna(-1)
        self.dataset["dport"] = self.dataset["dport"].apply(hex_to_dec)
        self.dataset["saddr"] = self.dataset["saddr"].apply(lambda x: ipv6_mac_to_int(x))
        self.dataset["daddr"] = self.dataset["daddr"].apply(lambda x: ipv6_mac_to_int(x))
        self.dataset["sport"] = self.dataset["sport"].apply(hex_to_dec)
        self.dataset["dport"] = self.dataset["dport"].apply(hex_to_dec)

        self.dataset["daddr"] = self.dataset["daddr"].str.replace('.', '')
        self.dataset["saddr"] = self.dataset["saddr"].str.replace('.', '')

    def make_train_test(self, sort_col, nptype, y_cols, balanced=False, test_ratio=0.2):
        '''
        make train data and test data that has appropriate normal-abnormal ratio
        :param sort_col: sort by value; str type
        :param nptype: ndarray type; str type
        :param y_cols: feature that shall be y not x; str type
        :param balanced: whether to balance the dataset; bool type
        :param test_ratio: ratio of test dataset; float type
        :return: train data and test data (x_train, x_test, y_train, y_test)
        '''
        df_y = self.dataset[y_cols]
        df_x = self.dataset.drop([y_cols], axis=1)

        if balanced:
            # Separate normal and abnormal data
            normal_data = self.dataset[self.dataset[y_cols] == 0]
            abnormal_data = self.dataset[self.dataset[y_cols] == 1]

            # Calculate the number of samples for each class based on the test ratio
            normal_test_size = int(len(normal_data) * test_ratio)
            abnormal_test_size = int(len(abnormal_data) * test_ratio)

            # Split the data into train and test sets for each class
            normal_train, normal_test = train_test_split(normal_data, test_size=normal_test_size, random_state=42)
            abnormal_train, abnormal_test = train_test_split(abnormal_data, test_size=abnormal_test_size, random_state=42)

            # Concatenate the train and test sets of each class
            df_x_train = pd.concat([normal_train, abnormal_train], axis=0)
            df_x_test = pd.concat([normal_test, abnormal_test], axis=0)
            df_y_train = df_x_train[y_cols]
            df_y_test = df_x_test[y_cols]
            df_x_train = df_x_train.drop([y_cols], axis=1)
            df_x_test = df_x_test.drop([y_cols], axis=1)

            if sort_col is not None:
                df_train_concat = pd.concat([df_x_train, df_y_train], axis=1)
                df_train_concat = df_train_concat.sort_values(by=sort_col)
                df_test_concat = pd.concat([df_x_test, df_y_test], axis=1)
                df_test_concat = df_test_concat.sort_values(by=sort_col)

                x_train = df_train_concat.drop([y_cols], axis=1).astype(nptype).to_numpy()
                y_train = df_train_concat[y_cols].astype(nptype).to_numpy()
                x_test = df_test_concat.drop([y_cols], axis=1).astype(nptype).to_numpy()
                y_test = df_test_concat[y_cols].astype(nptype).to_numpy()
            else:
                x_train = df_x_train.astype(nptype).to_numpy()
                x_test = df_x_test.astype(nptype).to_numpy()
                y_train = df_y_train.astype(nptype).to_numpy()
                y_test = df_y_test.astype(nptype).to_numpy()
        else:
            # stratify argument mostly preserves the normal-abnormal ratio in x and y
            df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x,
                                                                            df_y,
                                                                            stratify=df_y,
                                                                            test_size=test_ratio,
                                                                            random_state=42)
            if sort_col is not None:
                df_train_concat = pd.concat([df_x_train, df_y_train], axis=1)
                df_train_concat = df_train_concat.sort_values(by=sort_col)
                df_test_concat = pd.concat([df_x_test, df_y_test], axis=1)
                df_test_concat = df_test_concat.sort_values(by=sort_col)

                x_train = df_train_concat.drop([y_cols], axis=1).astype(nptype).to_numpy()
                y_train = df_train_concat[y_cols].astype(nptype).to_numpy()
                x_test = df_test_concat.drop([y_cols], axis=1).astype(nptype).to_numpy()
                y_test = df_test_concat[y_cols].astype(nptype).to_numpy()
            else:
                x_train = df_x_train.astype(nptype).to_numpy()
                x_test = df_x_test.astype(nptype).to_numpy()
                y_train = df_y_train.astype(nptype).to_numpy()
                y_test = df_y_test.astype(nptype).to_numpy()

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test, y_train, y_test

    def feature_selection_backward_elimination(self, i):
        X = self.dataset.drop('attack', axis=1)
        y = self.dataset['attack']

        lr = LinearRegression()
        # RFE 객체 생성, n_features_to_select는 선택할 특성 개수
        # 시계열 데이터의 특성을 유지하기 위해 33 + i로 설정
        rfe = RFE(estimator=lr, n_features_to_select=33 + i, step=1)  # 45는 state,proto,flg포함 18개를 뽑아낸다

        # 후진제거 적용
        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_].tolist()

        f = open("./selected_features.txt", "a")
        f.write(str(selected_features) + "\n")
        f.close()

        self.dataset = self.dataset[selected_features + ['attack']]

    def feature_selection_random_forest(self, x_train, y_train, feature_count):
        rf_clf = RandomForestClassifier(n_estimators=340)
        rf_clf.fit(x_train, y_train)

        column_count = self.dataset.shape[1]
        importances = rf_clf.feature_importances_
        importances_df = pd.DataFrame({"Features": self.dataset.iloc[:,[0, column_count]],
                                       "Importances": importances})
        importances_df.set_index("Importances")
        importances_df = importances_df.sort_values("Importances", ascending=False)
        return importances_df.iloc[:, [0, feature_count]]

    def feature_selection(self, feature_selection_method, x_train, y_train, feature_count):
        if feature_selection_method == "BackwardElimination":
            self.feature_selection_backward_elimination(feature_count)
        elif feature_selection_method == "RandomForest":
            self.feature_selection_random_forest(x_train, y_train, feature_count)
        else:
            self.feature_selection_backward_elimination(feature_count)