import os
import glob
import ntpath
import pandas as pd
import joblib
import numpy as np

from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, plot_roc_curve
from utils.folder_file_manager import save_file, make_directory_if_not_exists
from settings import MODEL_DIR, TRAINING_DIR, OUTPUT_DIR, PARAMETERS, PARAMETER_STATUS


class ClassifierTrainer:
    def __init__(self, para_idx):
        self.x_data = []
        self.y_data = []
        self.parameter_status = PARAMETER_STATUS[para_idx]

        self.model_names = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", ]
        self.classifiers = [
            KNeighborsClassifier(algorithm=PARAMETERS["KNN"]["algorithm"][para_idx],
                                 weights=PARAMETERS["KNN"]["weights"][para_idx],
                                 n_neighbors=PARAMETERS["KNN"]["n_neighbors"][para_idx]),
            SVC(kernel=PARAMETERS["SVM"]["kernel"][para_idx], C=PARAMETERS["SVM"]["C"][para_idx],
                gamma=PARAMETERS["SVM"]["gamma"][para_idx], degree=3, probability=True),
            DecisionTreeClassifier(criterion=PARAMETERS["DTC"]["criterion"][para_idx],
                                   splitter=PARAMETERS["DTC"]["splitter"][para_idx]),
            RandomForestClassifier(criterion=PARAMETERS["RFC"]["criterion"][para_idx],
                                   n_estimators=PARAMETERS["RFC"]["n_estimators"][para_idx],
                                   max_depth=PARAMETERS["RFC"]["max_depth"][para_idx]),
            GaussianNB(var_smoothing=PARAMETERS["GNB"]["var_smoothing"][para_idx])
        ]

    @staticmethod
    def convert_str_array(array_str):
        list_str = array_str.replace("  ", " ").replace(" ", ",")
        last_comma = list_str.rfind(",")
        f_list_str = list_str[:last_comma] + list_str[last_comma + 1:]
        converted_array = np.array(literal_eval(f_list_str))

        return converted_array

    def train_models(self, data_type, data_size, test_ratio, x_train, y_train, x_test, y_test):
        sub_output_dir = make_directory_if_not_exists(
            os.path.join(OUTPUT_DIR, f"{data_type}_{data_size}_{1 - test_ratio}:{test_ratio}_{self.parameter_status}"))
        training_report = {"accuracy": "", "f1-score": "", "confusion matrix": "", "precision": ""}
        for name, clf in zip(self.model_names, self.classifiers):
            model_path = os.path.join(MODEL_DIR, f'{data_type}_{data_size}_{1 - test_ratio}:{test_ratio}_{name}.clf')
            print(f"[INFO] Training {name} model with {data_type}, {data_size}, {1 - test_ratio}:{test_ratio} Data...")
            clf.fit(x_train, y_train)
            y_predicts = clf.predict(x_test)
            training_report["accuracy"] = accuracy_score(y_true=y_test, y_pred=y_predicts)
            training_report["f1-score"] = f1_score(y_true=y_test, y_pred=y_predicts)
            training_report["confusion matrix"] = confusion_matrix(y_true=y_test, y_pred=y_predicts)
            training_report["precision"] = [precision_score(y_pred=y_predicts, y_true=y_test),
                                            recall_score(y_true=y_test, y_pred=y_predicts)]
            save_file(content=str(training_report),
                      filename=os.path.join(sub_output_dir, f'{name}_report.txt'), method="w")
            fig, ax = plt.subplots()
            roc_curve = plot_roc_curve(clf, x_test, y_test, ax=ax)
            roc_curve.ax_.set_title(f"{data_type}_{data_size}_{1 - test_ratio}:{test_ratio}_{name} ROC Curve")
            # plt.show()
            fig.savefig(os.path.join(sub_output_dir, f'{name}_roc_curve.png'))
            joblib.dump(clf, model_path)
            print(f"[INFO] Trained {name} model with {data_type}, {data_size}, {1 - test_ratio}:{test_ratio} Data...")

        return

    def train(self):
        train_data_paths = glob.glob(os.path.join(TRAINING_DIR, 'data', "*.csv"))
        for t_data_path in train_data_paths:
            file_name = ntpath.basename(t_data_path).replace(".csv", "")
            data_type = file_name.split("_")[0]
            data_size = file_name.split("_")[1]
            print(f"[INFO] Dataset importing {t_data_path}...")
            train_df = pd.read_csv(t_data_path, index_col=0)
            train_df_x = train_df.drop('label', axis=1).values.tolist()
            train_df_y = train_df['label'].values.tolist()
            for test_ratio in [0.1, 0.3, 0.4]:
                x_train, x_test, y_train, y_test = \
                    train_test_split(train_df_x, train_df_y, test_size=test_ratio, random_state=42)
                self.train_models(data_type=data_type, data_size=data_size, test_ratio=test_ratio, x_train=x_train,
                                  y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    ClassifierTrainer(para_idx=0).train()
