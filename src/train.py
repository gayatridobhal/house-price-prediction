#model training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from joblib import dump, load
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'train.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'test.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'ridge_model.joblib')

train = pd.read_csv(TRAIN_DATA_PATH)
y = np.log1p(train['SalePrice'])
#House proces are right skewed
#X = train.select_dtypes(exclude=['object']).drop(['SalePrice', 'Id'],axis=1).fillna(0)
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt"
]
X = train[FEATURES].fillna(0)
model = Ridge(alpha=10)
kf = KFold(5, shuffle = True, random_state = 42)
#k-fold cross validation
score = np.sqrt(-cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = kf))
print("Ridge CV RMSE:", score.mean())
#0.15
model.fit(X,y)
dump(model, MODEL_PATH)

model = load(MODEL_PATH)
test = pd.read_csv(TEST_DATA_PATH)
#X_test = test.select_dtypes(exclude = ['object']).drop(['Id'],axis=1).fillna(0)
X_test = test[FEATURES].fillna(0)
preds = np.expm1(model.predict(X_test))
#back to original scale

submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds})
submission.to_csv("submission.csv", index = False)
