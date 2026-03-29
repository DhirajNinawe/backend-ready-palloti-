import pandas as pd

# LOAD DATA
diabetes = pd.read_csv("data/diabetes.csv")
heart = pd.read_csv("data/heart.csv")
asthma = pd.read_csv("data/asthma.csv")
tb = pd.read_csv("data/tuberculosis.csv")

# ================= STANDARDIZE =================

def prepare(df, disease_name):
    
    df = df.copy()

    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # Create standard features
    data = pd.DataFrame()

    data["age"] = df.get("age", 0)
    data["glucose"] = df.get("glucose", df.get("blood_sugar", 0))
    data["bmi"] = df.get("bmi", 0)
    data["cholesterol"] = df.get("chol", df.get("cholesterol", 0))
    data["smoking"] = df.get("smoking", 0)

    # Disease labels
    data["diabetes"] = 1 if disease_name == "diabetes" else 0
    data["heart"] = 1 if disease_name == "heart" else 0
    data["asthma"] = 1 if disease_name == "asthma" else 0
    data["tuberculosis"] = 1 if disease_name == "tb" else 0

    return data

# APPLY
d1 = prepare(diabetes, "diabetes")
d2 = prepare(heart, "heart")
d3 = prepare(asthma, "asthma")
d4 = prepare(tb, "tb")

# COMBINE
data = pd.concat([d1, d2, d3, d4])

# SAVE
data.to_csv("data/combined.csv", index=False)

print("✅ Combined dataset ready")