import os

from utils.folder_file_manager import make_directory_if_not_exists

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(CUR_DIR, 'training_data')
OUTPUT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'output'))
MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model'))

SMALL = 717
MEDIUM = 1433

PARAMETER_STATUS = ["before", "after"]
PARAMETERS = {"KNN": {"algorithm": ["ball_tree", "kd_tree"], "weights": ["uniform", "distance"], "n_neighbors": [3, 5]},
              "SVM": {"C": [1.0, 2.0], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
              "DTC": {"criterion": ["gini", "entropy"], "splitter": ["best", "random"]},
              "RFC": {"criterion": ["gini", "entropy"], "n_estimators": [10, 100], "max_depth": [None, 5]},
              "GNB": {"var_smoothing": [1e-9, 5e-9]}}

BAD_WORD_FILE_PATH = ""
DATA_SAMPLES = 10000
