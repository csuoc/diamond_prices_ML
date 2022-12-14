import warnings
warnings.filterwarnings("ignore")

#Read
import pandas as pd
train = pd.read_csv("data/train.csv")
    
#Drop useless columns
train.drop(["depth", "table"], axis=1, inplace=True)
    
# Label encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
columns = ["cut", "color", "clarity"]
for i in columns:
    train[i] = le.fit_transform(train[i])
    train_noprice = train.drop(["id", "price"], axis=1)

# Defining Variables (this time no split, all my dataset will be the training data)
X_train = train_noprice
y_train = train["price"]

# Checking
try:
    X_train.shape[0] == y_train.shape[0]
except:
    print("Something went wrong when splitting")
        
# Defining model

from catboost import CatBoostRegressor
    
model = CatBoostRegressor(loss_function="RMSE", depth = 6, learning_rate = 0.1, iterations = 900, l2_leaf_reg = 0.25, random_strength=10, grow_policy="SymmetricTree")
    
# Fitting

model.fit(X_train, y_train)
    
# Save model
    
import pickle
pickle.dump(model, open("models/my_model", 'wb'))