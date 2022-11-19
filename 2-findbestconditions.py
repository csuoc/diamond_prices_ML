print("TEST")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
# Checking
try:
     X_train.shape[0] == y_train.shape[0]
except:
    print("Something went wrong when splitting")
        
# Importing model and parameters

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

model_CBR = CatBoostRegressor()

parameters = {'loss_function':["RMSE"],
              'depth' : [5,6,7,8],
              'learning_rate' : [0.075, 0.1, 0.3],
              'iterations'    : [600,700,800,1000],
              'l2_leaf_reg': [0.25, 0.5, 1],
              'bagging_temperature': [0, 1],
              'random_strength':[5,10,15]
             }

grid = GridSearchCV(estimator=model_CBR, param_grid = parameters, cv = 3, n_jobs=-1)

# Fitting

grid.fit(X_train, y_train)

# Best estimator

print("Results from Grid Search")
print("The best estimator across ALL searched params: ", grid.best_estimator_)
print("The best score across ALL searched params: ", grid.best_score_)
print("The best parameters across ALL searched params: ", grid.best_params_)