import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor  # Use regressor for numeric scores
import pickle
import numpy as np

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset/jobs.csv")

# Example: assume 'skills' column is text
# If your dataset doesn't have numeric demand scores, we create dummy scores for demo
# Later you can replace with real scores based on skill demand
df['demand_score'] = np.random.rand(len(df))  # Random score between 0 and 1

X_text = df["skills"].str.lower()  # Features
y = df["demand_score"]             # Numeric target

# ---------------- VECTORIZE ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)  # FIT vectorizer

# ---------------- TRAIN MODEL ----------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- SAVE MODEL + VECTORIZER ----------------
import os
if not os.path.exists("model"):
    os.makedirs("model")

with open("model/skill_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("✅ Model and vectorizer saved successfully!")
