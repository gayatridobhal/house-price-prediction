#data cleaning and feature engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'train.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'test.csv')

def load_data():
    train = pd.read_csv(TRAIN_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)
    return train, test

def clean_data(df):
    df['LotFrontage'].fillna(df['LotFrontage'].median, inplace=True)
    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)
    for col in ['Alley', 'PoolQC', 'Fence', 'MiscFeature']:
        df[col] = df[col].fillna('None')
    return df

def feature_engineering(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    return df

train,test = load_data()
train = clean_data(train)
train = feature_engineering(train)