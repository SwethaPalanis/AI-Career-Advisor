# ----------------------------
# AI Career Advisor - Streamlit App (Ready-to-Run)
# ----------------------------

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ----------------------------
# Session State Init
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'login_clicked' not in st.session_state:
    st.session_state.login_clicked = False

# ----------------------------
# Load model & vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    with open("model/skill_model.pkl", "rb") as f:
        return pickle.load(f)

model, vectorizer = load_model()

# ----------------------------
# Helper functions
# ----------------------------
def predict_roles(user_skills):
    """
    Input: list of user skills
    Output: predicted role & recommended skills
    """
    # Convert input to string
    skill_str = ",".join(user_skills).lower()
    
    # Vectorize
    user_vector = vectorizer.transform([skill_str])
    
    # Predict probabilities
    predictions = model.predict_proba(user_vector)
    role_names = model.classes_
    predicted_role = role_names[predictions.argmax()]
    
    # Example recommended skills logic
    recommended = [
        ("Python", 95),
        ("SQL", 80),
        ("Machine Learning", 70),
        ("Communication", 50),
        ("React", 60)
    ]
    
    return predicted_role, recommended

# ----------------------------
# LOGIN PAGE
# ----------------------------
def login():
    st.title("🔐 AI Career Advisor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state.login_clicked = True
        if username == "user" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()  # Correct way
        else:
            st.error("❌ Invalid username or password")
    
    # Only show error if button clicked
    if st.session_state.login_clicked and (username != "user" or password != "1234"):
        st.error("❌ Invalid username or password")

# ----------------------------
# LOGOUT FUNCTION
# ----------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()

# ----------------------------
# MAIN APP
# ----------------------------
def main():
    # Sidebar
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    st.sidebar.write("AI Career Advisor Dashboard")
    st.sidebar.button("Logout", on_click=logout)
    
    st.title("💼 AI Career Advisor")
    st.subheader("Enter your skills (comma separated):")
    user_input = st.text_input("Ex: Python, SQL, Machine Learning")
    
    if st.button("Get Recommended Roles"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter at least one skill")
        else:
            user_skills = [s.strip() for s in user_input.split(",")]
            predicted_role, recommended = predict_roles(user_skills)
            
            # Show predicted role
            st.success(f"✅ Predicted Role: {predicted_role}")
            
            # Recommended Skills Graph (1-10 scale)
            st.subheader("📊 Recommended Skills to Learn")
            st.write("Demand scores are normalized to 1–10 scale.")

            skills = [skill.upper() for skill, score in recommended]
            scores = [score for skill, score in recommended]

            max_score = max(scores)
            scores_1_10 = [round(1 + (s / max_score) * 9, 1) for s in scores]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(skills, scores_1_10, color="#00ffcc")

            ax.set_xlabel("Demand Score (1-10)")
            ax.set_title("Skill Trend / Future Demand")

            for i, v in enumerate(scores_1_10):
                ax.text(v + 0.1, i, str(v), color='black', va='center')

            ax.invert_yaxis()
            st.pyplot(fig)

# ----------------------------
# APP FLOW CONTROL
# ----------------------------
if not st.session_state.logged_in:
    login()
else:
    main()
