import math

import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def clean(df):
    df=df.drop(df[df.resprate>=60].index)
    df=df.drop(df[(df.o2sat<50) | (df.o2sat>101)].index)
    df=df.drop(df[(df.map<0) | (df.map>180)].index)
    return df

def normalVal(x,attr):
    if attr in ['heartrate','resprate','map']:
        return 0
    return 1-x


train_df = pd.read_csv("../dataset/train.csv")
test_df = pd.read_csv('../dataset/test.csv')

# train_df = train_df.head(100)
print(train_df.shape)

train_df = train_df.dropna().reset_index(drop=True)

print(train_df.shape)

upper_bounds = {'heartrate': 100, 'resprate': 24, 'o2sat': 100, 'map': 105}
lower_bounds = {'heartrate': 60, 'resprate': 12, 'o2sat': 95, 'map': 70}

for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
    print(min(train_df[attribute]), max(train_df[attribute]))
# print(train_df['o2sat'])
# print(train_df['o2sat'][2])

# for df in [train_df, test_df]:
#     df=df.drop(df[df.resprate>=60].index)
#     df=df.drop(df[(df.o2sat<50) | (df.o2sat>101)].index)
#     df=df.drop(df[(df.map<0) | (df.map>200)].index)

# train_df=train_df.drop(train_df[(train_df.map<0) | (train_df.map>200)].index)
train_df = clean(train_df)
test_df = clean(test_df)

for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
    print(min(train_df[attribute]), max(train_df[attribute]))

# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     for df in [train_df, test_df]:
#         df[attribute] = df[attribute].transform(lambda x: ( 0 if (0 <= x <= 1.0) else min([abs(x),abs(x-1)]) ))

for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
    for df in [train_df, test_df]:
        df[attribute+'_err'] = (df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])


for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
    for df in [train_df, test_df]:
        df[attribute+'_err'] = df[attribute+'_err'].transform(
            lambda x: ( 0.1*normalVal(x,attribute) if (0 <= x <= 1.0) else 0.1+(-x if x<0 else x-1) ))
        df[attribute] = np.abs(df[attribute] - (lower_bounds[attribute]+upper_bounds[attribute])/2.0)
        df[attribute] = np.divide(df[attribute],np.max(df[attribute]))
        # plt.hist(df[attribute+'_err'],bins=100)
        # plt.title(attribute+'_err')
        # plt.show()

print(train_df.shape)
print(test_df.shape)



# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     train_df[attribute] = (train_df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])
# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     test_df[attribute] = (test_df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])

train_df['time'] = pd.to_datetime(train_df['time']).apply(lambda x: int(int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))))

test_df['time'] = pd.to_datetime(test_df['time']).apply(lambda x: int(int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))))

previous_id = None
previous_time = 0
for df in [train_df, test_df]:
    for index, row in df.iterrows():
            current_id = row['id']
            current_time = row['time']
            if current_id != previous_id:
                df.at[index, 'time'] = 0
                previous_time = current_time
            else:
                df.at[index, 'time'] = (current_time - previous_time) / 86400
            previous_id = current_id

previous_id = None
previous_v = 0
for attr in ['heartrate', 'resprate', 'o2sat', 'map']:
    for df in [train_df, test_df]:
        for index, row in df.iterrows():
                current_id = row['id']
                current_v = row[attr]
                if current_id != previous_id:
                    df.at[index, attr+"_delta"] = 0
                    previous_v = current_v
                else:
                    df.at[index, attr+"_delta"] = (current_v - previous_v)
                previous_id = current_id




df0=train_df[train_df.label==0]
df1=train_df[train_df.label==1]
print(df0.shape,df1.shape)
# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     for df in [df0, df1]:
#         plt.hist(df[attribute+'_err'],bins=50)
#         plt.title(attribute+'_err')
#         plt.show()
for attribute in ['heartrate_delta', 'resprate_delta', 'o2sat_delta', 'map_delta']:
    for df in [df0, df1]:
        plt.hist(df[attribute], bins=50)
        plt.title(attribute)
        plt.show()


train_df.to_csv("../dataset/train_clean.csv")
test_df.to_csv("../dataset/test_clean.csv")
