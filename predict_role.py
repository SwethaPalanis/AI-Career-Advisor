import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset/jobs.csv")

# Prepare role-skill corpus
role_skills = df["skills"].str.lower().tolist()
roles = df["job_title"].tolist()

# Vectorize skills
vectorizer = TfidfVectorizer()
role_vectors = vectorizer.fit_transform(role_skills)

# ---------- USER INPUT ----------
user_input = input("Enter your skills (comma separated): ").lower()

user_vector = vectorizer.transform([user_input])

# Similarity calculation
similarities = cosine_similarity(user_vector, role_vectors)[0]

# Rank roles
role_scores = list(zip(roles, similarities))
role_scores = sorted(role_scores, key=lambda x: x[1], reverse=True)

# ---------- OUTPUT ----------
print("\n🎯 BEST MATCHING IT ROLES FOR YOU:\n")

for role, score in role_scores[:3]:
    print(f"• {role}  → Match Score: {round(score, 3)}")
