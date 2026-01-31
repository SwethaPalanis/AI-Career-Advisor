import pickle

# Load trained model + vectorizer
with open("model/skill_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

print("✅ Model loaded successfully")

# ---------- MANUAL INPUT ----------
# Neenga inga skills change pannalaam
user_skills = input("Enter skills (comma separated): ")

# Example input:
# python, ml, genai, llm

# Convert input to model format
skill_text = [user_skills]

X_test = vectorizer.transform(skill_text)
