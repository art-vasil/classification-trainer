import pandas as pd
import random
from tqdm import trange
from random import sample


bad_words = pd.read_excel("/media/main/Data/Task/ClassificationTrainer/Compairing/Bad words.xlsx")['words'].astype('str').tolist()
thrshld = 10

data_arr = []
def add_samples(count, start, end):
  """
  Create dataset by using the features provided in the xlsx file
  """
  for i in trange(count):
    num_features = random.randint(start, end)   		 # generate a number between an interval 
    selctd_features = sample(bad_words, num_features)    # get random number of features based on num_features
    feature_vect = {feature:1 for feature in selctd_features}  # set the selectd_features to 1
    data_arr.append(feature_vect)

add_samples(50000, 0, thrshld-1)					# create positive features. i.e. number of bad words < 10
add_samples(10000, thrshld, len(bad_words))		# create positive features. i.e. number of bad words > 10
data = pd.DataFrame(data_arr, columns = bad_words)
data.fillna(0, inplace=True)


data['__label__'] =  data.sum(axis = 1) >= thrshld
data.to_csv("dataset.csv")