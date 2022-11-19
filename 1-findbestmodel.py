import warnings
warnings.filterwarnings("ignore")

#Read
import pandas as pd
df = pd.read_csv("data/train.csv")
    
#Drop useless columns
df.drop(["depth", "table", "x", "y", "z"], axis=1, inplace=True)
    
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
        
# Defining model
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
    
models = {
"lr": LinReg(),
"ridge": Ridge(),
"lasso": Lasso(),
"sgd": SGDRegressor(),
"knn": KNeighborsRegressor(),
"grad": GradientBoostingRegressor(),
"svr": SVR(),
"randomregressor": RandomForestRegressor(),
"decisiontree": DecisionTreeRegressor(),
"catboost": CatBoostRegressor(loss_function="RMSE"),
"xgboost": XGBRegressor()
}
    
# Fitting
for name, model in models.items():
    print("Fitting: ", name)
    model.fit(X_train, y_train)
    
# Get errors
from sklearn import metrics
import numpy as np

for name, model in models.items():
    y_pred = model.predict(X_test)

    # Variables
    print(f"------{name}------")
    print('MAE - ', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE - ', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE - ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 - ', metrics.r2_score(y_test, y_pred))