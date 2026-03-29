import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(path, target_col, save_name):

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if target_col.lower() not in df.columns:
        print(f"❌ Target not found in {path}")
        print("Columns:", df.columns)
        return

    # 👉 REMOVE NON-NUMERIC
    df = df.select_dtypes(include=['number'])

    if target_col.lower() not in df.columns:
        print(f"❌ Target removed after cleaning in {path}")
        return

    X = df.drop(target_col.lower(), axis=1)
    y = df[target_col.lower()]

    model = RandomForestClassifier()
    model.fit(X, y)

    pickle.dump(model, open(save_name, "wb"))
    print(f"✅ {save_name} saved")


# ================= TRAIN =================

train_model("data/diabetes.csv", "Outcome", "diabetes_model.pkl")
train_model("data/heart.csv", "target", "heart_model.pkl")
train_model("data/asthma.csv", "diagnosis", "asthma_model.pkl")