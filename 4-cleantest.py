#Read
import pandas as pd
df = pd.read_csv("data/test.csv")
    
#Drop useless columns
df.drop(["depth", "table"], axis=1, inplace=True)
    
# Label encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
columns = ["cut", "color", "clarity"]
for i in columns:
    df[i] = le.fit_transform(df[i])
    df_clean = df.drop(["id"], axis=1)

df_clean.to_csv("data/testclean.csv", index=False)