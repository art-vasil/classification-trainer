# ClassificationTrainer

## Overview
This project is to generate the 2 types of random dataset and train the 5 classification models with the generated dataset
to estimate the model with f1-score, precision-recall score, confusion matrix, accuracy.
Scikit-learn, Pandas, Matplotlib libraries are used in this project.

## Structure

- src

    The main source code for generating the dataset and training the models

- utils

    * The trained models
    * The source code for the management of folders and files of this project
    
- training_data

    * Bad words files with 3 types of sizes.
    * The 2 types of dataset based on the bad word files
    
- app

    The main execution file
    
- requirements

    All the dependencies for this project
    
- settings

    The several settings including file path
    
## Installation

- Environment

    Ubuntu 18.04, Windows 10, Python 3.6

- Dependency Installation

    Please navigate to the project directory and run the following command in the terminal
    ```
        pip3 install -r requirements.txt
    ``` 

## Execution

- If you want to generate the training dataset, please set BAD_WORD_FILE_PATH in settings with the absolute path of 
original bad words excel file and DATA_SAMPLES with the number of dataset as you need. 
Then please run the following command in the terminal.

    ```
        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
        python3 src/data/generator.py
    ```

- If you want to estimate the model with the generated dataset, please run the following command in the terminal.

    ```
        python3 app.py
    ```

The result of model training will be in the output directory.
