import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# LOAD
data = pd.read_csv("data/combined.csv")

# FEATURES
X = data[["age", "glucose", "bmi", "cholesterol", "smoking"]]

# TARGET (multi-label)
y = data[["diabetes", "heart", "asthma", "tuberculosis"]]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL
model = RandomForestClassifier()
model.fit(X_train, y_train)

# SAVE
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully")