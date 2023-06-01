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


FIX_LEN = 20
BASE_ATTR = ['heartrate','resprate','map', 'o2sat']
BASE_ATTR_ERR = ['heartrate_err','resprate_err','map_err', 'o2sat_err']
BASE_ATTR_DELTA = ['heartrate_delta','resprate_delta','map_delta', 'o2sat_delta']
def toDfFixLen(df):
    group = df.groupby('id')
    df24 = pd.DataFrame([], columns=colFixLen, dtype=float)
    for key, value in group:
        # print('value.shape',value.shape)
        # print('value.columns',value.columns)
        # print('value',value)
        patient = [value.iloc[-1]['id']]
        lastTmp = 0
        for i in range(FIX_LEN):
            tmp = (i + value.shape[0] - FIX_LEN) if value.shape[0] >= FIX_LEN else min(i, value.shape[0] - 1)
            line = value.iloc[tmp]
            lastLine = line if i==0 else value.iloc[lastTmp]
            # print('line',line)
            # print('\ttmp\t',tmp)
            for col in BASE_ATTR+BASE_ATTR_ERR:
                patient.append(line[col])
            for attr_id in range(len(BASE_ATTR)):
                attr = BASE_ATTR[attr_id]
                patient.append(line[attr]-lastLine[attr])
            lastTmp = tmp
        patient.append(value.iloc[-1]['label']) # label
        # print(len(patient), patient)
        # print(value.shape)
        # df24=df24.append(patient,ignore_index=True)
        df24.loc[len(df24.index)] = patient
        if df24.shape[0]%1000==0:
            print('\tturning to dfFixLen. shape:\t',df24.shape)
    print('\tTurn to dfFixLen over. shape:\t', df24.shape)
    return df24


train_df = pd.read_csv("./dataset/train.csv")
test_df = pd.read_csv('./dataset/test.csv')

# train_df = train_df.head(100)
print(train_df.shape)

train_df = train_df.dropna().reset_index(drop=True)

print(train_df.shape)

upper_bounds = {'heartrate': 100, 'resprate': 24, 'o2sat': 100, 'map': 105}
lower_bounds = {'heartrate': 60, 'resprate': 12, 'o2sat': 95, 'map': 70}
default_values = {'heartrate': 75, 'resprate': 20, 'o2sat': 100, 'map': 80}

for attribute in ['heartrate', 'resprate', 'map', 'o2sat']:
    print(min(train_df[attribute]), max(train_df[attribute]))

train_df = clean(train_df)
test_df = clean(test_df)

for attribute in ['heartrate', 'resprate', 'map', 'o2sat']:
    print(min(train_df[attribute]), max(train_df[attribute]))

# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     for df in [train_df, test_df]:
#         df[attribute] = df[attribute].transform(lambda x: ( 0 if (0 <= x <= 1.0) else min([abs(x),abs(x-1)]) ))

for attribute in ['heartrate', 'resprate', 'map', 'o2sat']:
    for df in [train_df, test_df]:
        df[attribute+'_err'] = (df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])



for attribute in ['heartrate', 'resprate', 'map', 'o2sat']:
    for df in [train_df, test_df]:
        df[attribute+'_err'] = df[attribute+'_err'].transform(
            lambda x: ( 0.1*normalVal(x,attribute) if (0 <= x <= 1.0) else 0.1+(-x if x<0 else x-1) ))
        print('???',min(df[attribute+'_err']),max(df[attribute+'_err']))
        df[attribute] = np.abs(df[attribute] - default_values[attribute])
        df[attribute] = np.divide(df[attribute],np.max(df[attribute]))
        # plt.hist(df[attribute+'_err'],bins=100)
        # plt.title(attribute+'_err')
        # plt.show()
df0 = train_df[train_df.label == 0]
df1 = train_df[train_df.label == 1]
for attribute in BASE_ATTR_ERR:
    for df in [df0, df1]:
        plt.hist(df[attribute], bins=50)
        plt.title("label=" + str(0 if df is df0 else 1) + ", " + attribute)
        plt.show()
print(train_df.shape)
print(test_df.shape)


df0=train_df[train_df.label==0]
df1=train_df[train_df.label==1]
print(df0.shape,df1.shape)
for attribute in BASE_ATTR:
    for df in [df0, df1]:
        plt.hist(df[attribute], bins=50)
        plt.title("label=" + str(0 if df is df0 else 1) + ", " + attribute)
        plt.show()

# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     train_df[attribute] = (train_df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])
# for attribute in ['heartrate', 'resprate', 'o2sat', 'map']:
#     test_df[attribute] = (test_df[attribute] - lower_bounds[attribute]) / (upper_bounds[attribute] - lower_bounds[attribute])

# train_df['time'] = pd.to_datetime(train_df['time']).apply(lambda x: int(int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))))
#
# test_df['time'] = pd.to_datetime(test_df['time']).apply(lambda x: int(int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))))
#
# previous_id = None
# previous_time = 0
# for df in [train_df, test_df]:
#     for index, row in df.iterrows():
#             current_id = row['id']
#             current_time = row['time']
#             if current_id != previous_id:
#                 df.at[index, 'time'] = 0
#                 previous_time = current_time
#             else:
#                 df.at[index, 'time'] = (current_time - previous_time) / 86400
#             previous_id = current_id
#
# previous_id = None
# previous_v = 0
# for attr in ['heartrate', 'resprate', 'o2sat', 'map']:
#     for df in [train_df, test_df]:
#         for index, row in df.iterrows():
#                 current_id = row['id']
#                 current_v = row[attr]
#                 if current_id != previous_id:
#                     df.at[index, attr+"_delta"] = 0
#                     previous_v = current_v
#                 else:
#                     df.at[index, attr+"_delta"] = (current_v - previous_v)
#                 previous_id = current_id



colFixLen = ['id']
for i in range(FIX_LEN):
    for attr in BASE_ATTR+BASE_ATTR_ERR+BASE_ATTR_DELTA:
        colFixLen.append('T'+str(i)+"_"+attr)
colFixLen.append('label')
print('colFixLen:\t', len(colFixLen), colFixLen)

train_dfFixLen = toDfFixLen(train_df)
test_dfFixLen = toDfFixLen(test_df)

# train_dfFixLen.to_csv("./dataset/train_clean"+str(FIX_LEN)+".csv")
# test_dfFixLen.to_csv("./dataset/test_clean"+str(FIX_LEN)+".csv")
df0=train_dfFixLen[train_dfFixLen.label==0]
df1=train_dfFixLen[train_dfFixLen.label==1]

for attribute in BASE_ATTR_DELTA:
    for df in [df0, df1]:
        val = []
        for tmp in ['T'+str(i)+"_"+attribute for i in range(FIX_LEN)]:
            val=val+df[tmp].tolist()
        plt.hist(val, bins=50)
        plt.title("label=" + str(0 if df is df0 else 1) + ", " + attribute)
        plt.show()

