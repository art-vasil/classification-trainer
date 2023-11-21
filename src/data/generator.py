import os
import numpy as np
import pandas as pd
import random

from tqdm import trange
from random import sample
from utils.folder_file_manager import make_directory_if_not_exists
from settings import BAD_WORD_FILE_PATH, DATA_SAMPLES, TRAINING_DIR, SMALL, MEDIUM


class DataGenerator:
    def __init__(self):
        self.bad_words = pd.read_excel(BAD_WORD_FILE_PATH)['words'].astype('str').tolist()
        self.threshold = 10

    @staticmethod
    def add_samples(data_arr, bad_words, start, end, sampling_number):
        """
        Create dataset by using the features provided in the xlsx file
        """
        for i in trange(sampling_number):
            i += 1
            num_features = random.randint(start, end)  # generate a number between an interval
            selected_features = sample(bad_words, num_features)  # get random number of features based on num_features
            feature_vect = {feature: 1 for feature in selected_features}  # set the selectd_features to 1
            data_arr.append(feature_vect)

        return data_arr

    @staticmethod
    def create_2nd_generation(bad_words, data_size):
        data_file_path = os.path.join(TRAINING_DIR, "data", f"Feature_{data_size}_data.csv")
        mu, sigma = 0, 1  # mean and standard deviation
        x_0 = np.random.normal(mu, sigma, (int(0.8 * DATA_SAMPLES), len(bad_words)))
        x_1 = np.random.normal(mu + 0.85, sigma * 0.50, (int(0.2 * DATA_SAMPLES), len(bad_words)))
        features = np.concatenate((x_0, x_1), axis=0)
        data_df = pd.DataFrame(features, columns=bad_words)
        data_df['label'] = np.array([True] * int(0.8 * DATA_SAMPLES) + [False] * int(0.2 * DATA_SAMPLES)).T
        # print(len(data_df[data_df["label"] == True]))
        data_df.to_csv(data_file_path)

        print(f"[INFO] Generated Feature_{data_size}_data into {data_file_path}")

        return

    def create_1st_data_generation(self, bad_words, data_size):
        data_file_path = os.path.join(TRAINING_DIR, "data", f"Random_{data_size}_data.csv")
        init_data = self.add_samples(data_arr=[], start=0, end=self.threshold - 1,
                                     sampling_number=int(0.8 * DATA_SAMPLES),
                                     bad_words=bad_words)
        # create positive features. i.e. number of bad words > 10
        final_data = self.add_samples(data_arr=init_data, sampling_number=int(0.2 * DATA_SAMPLES), start=self.threshold,
                                      end=len(bad_words), bad_words=bad_words)
        data_df = pd.DataFrame(final_data, columns=bad_words)
        data_df.fillna(0, inplace=True)

        data_df['label'] = data_df.sum(axis=1) >= self.threshold
        data_df.to_csv(data_file_path)

        print(f"[INFO] Generated Random_{data_size}_data into {data_file_path}")

        return

    def run(self):
        make_directory_if_not_exists(os.path.join(TRAINING_DIR, "data"))
        make_directory_if_not_exists(os.path.join(TRAINING_DIR, "bad_words"))
        small_bad_words = random.sample(self.bad_words, SMALL)
        medium_bad_words = random.sample(self.bad_words, MEDIUM)
        for b_words, b_size in zip([small_bad_words, medium_bad_words, self.bad_words], ["Small", "Medium", "Full"]):
            b_words_path = os.path.join(TRAINING_DIR, "bad_words", f"{b_size}_bad_words.csv")
            pd.DataFrame(np.array(b_words).T, columns=["words"]).to_csv(b_words_path, index=False)
            print(f"[INFO] Saved {b_size} Bad Words into {b_words_path}")
            self.create_1st_data_generation(bad_words=b_words, data_size=b_size)
            self.create_2nd_generation(bad_words=b_words, data_size=b_size)

        return


if __name__ == '__main__':
    DataGenerator().run()
