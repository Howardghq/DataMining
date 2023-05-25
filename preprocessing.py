import pandas as pd
import time
from datetime import datetime

train_df = pd.read_csv("../dataset/train.csv")
test_df = pd.read_csv('../dataset/test.csv')

# train_df = train_df.head(100)
print(train_df.shape)

train_df = train_df.dropna().reset_index(drop=True)

print(train_df.shape)

upper_bounds = {'heartrate': 100, 'resprate': 24, 'o2sat': 100, 'map': 105}
lower_bounds = {'heartrate': 60, 'resprate': 12, 'o2sat': 95, 'map': 70}

for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
    train_df[attribute] = (train_df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])

for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
    test_df[attribute] = (test_df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])

train_df['time'] = pd.to_datetime(train_df['time']).apply(lambda x: int(int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))))

test_df['time'] = pd.to_datetime(test_df['time']).apply(lambda x: int(int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))))

train_df.to_csv("../dataset/train_clean.csv")
test_df.to_csv("../dataset/test_clean.csv")
