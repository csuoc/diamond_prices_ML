import pandas as pd
import pickle

# Read
testclean = pd.read_csv("data/testclean.csv")

# Load model
loaded_model = pickle.load(open ("models/my_model", "rb"))

# Predict
y_pred = loaded_model.predict(testclean)

# Adding ID column
test = pd.read_csv("data/test.csv")
testclean["id"] = test["id"]

# Adding y_pred column
testclean['price'] = y_pred

# Modifying for submission
test_for_submission = testclean[["id", "price"]]

# Save
test_for_submission.to_csv("my_submission.csv", index=False)