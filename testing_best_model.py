import warnings
warnings.filterwarnings("ignore")

#Read
import pandas as pd
df = pd.read_csv("data/train.csv")
 
#Drop useless columns
df.drop(["depth", "table"], axis=1, inplace=True)
    
# Label encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
columns = ["cut", "color", "clarity"]
for i in columns:
    df[i] = le.fit_transform(df[i])
    df_noprice = df.drop(["id", "price"], axis=1)

# Defining Variables
X = df_noprice
y = df["price"]
    
# Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
# Checking
try:
     X_train.shape[0] == y_train.shape[0]
except:
    print("Something went wrong when splitting")
        
# Importing model and parameters

from catboost import CatBoostRegressor

model_CBR = CatBoostRegressor(loss_function="RMSE", depth = 6, learning_rate = 0.1, iterations = 900, l2_leaf_reg = 0.25, random_strength=10, grow_policy="SymmetricTree")

# Fitting

model_CBR.fit(X_train, y_train)


# Get errors
from sklearn import metrics
import numpy as np

y_pred = model_CBR.predict(X_test)

# Variables
print("------CatBoost results------")
print('MAE - ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE - ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE - ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 - ', metrics.r2_score(y_test, y_pred))