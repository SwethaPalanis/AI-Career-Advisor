import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("dataset/jobs.csv")

# Load model
with open("model/skill_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# User input
user_input = input("Enter your current skills (comma separated): ")
user_input = user_input.lower()

user_skills = set([s.strip() for s in user_input.split(",")])

# Extract all skills
all_skills = set()
for skills in df["skills"]:
    for s in skills.split(","):
        all_skills.add(s.strip().lower())

# Missing skills
missing_skills = list(all_skills - user_skills)

skill_scores = {}

for skill in missing_skills:
    # 🔥 CONTEXT-AWARE INPUT
    combined_text = user_input + ", " + skill

    X_test = vectorizer.transform([combined_text])
    score = model.predict(X_test)[0]

    skill_scores[skill] = score

# Sort
recommended = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)

print("\n🔥 CONTEXT-AWARE SKILL RECOMMENDATIONS:\n")
for skill, score in recommended[:5]:
    print(f"• {skill.upper()}  → Score: {round(score, 3)}")
