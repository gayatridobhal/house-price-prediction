#unit test
from src.preprocess import clean_data
import pandas as pd

def test_missing_values_handled():
    df = pd.DataFrame({'LotFrontage': [None, 50]})
    df2 = clean_data(df)
    assert df2['LotFrontage'].isnull().sum() == 0
