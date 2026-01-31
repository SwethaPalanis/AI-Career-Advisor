import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import os
import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Career Advisor", page_icon="🤖", layout="centered")

# ---------------- DARK MODE CSS ----------------
st.markdown("""
<style>
body {background-color: #121212; color: #e0e0e0;}
.main-title {text-align:center; font-size:40px; font-weight:700; color:#ffcc00;}
.subtitle {text-align:center; font-size:18px; color:#cccccc; margin-bottom:30px;}
.card {background-color:#1e1e1e; padding:20px; border-radius:12px; box-shadow:0px 4px 12px rgba(0,0,0,0.5); margin-bottom:20px;}
.footer {text-align:center; color:gray; margin-top:40px;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🤖 AI Career Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Job Role & Skill Recommendation System</div>', unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.subheader("🔑 Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        users = pd.read_csv("dataset/users.csv")
        user_match = users[(users['username'] == username_input) & (users['password'] == password_input)]
        if not user_match.empty:
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.success(f"Welcome {username_input}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# ---------------- MAIN APP ----------------
else:
    username = st.session_state.username

    # ---------------- LOGOUT BUTTON ----------------
    if st.sidebar.button("🔒 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    # ---------------- LOAD DATA + MODEL ----------------
    df = pd.read_csv("dataset/jobs.csv")
    with open("model/skill_model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)

    # ---------------- SIDEBAR INPUT ----------------
    st.sidebar.header("🧠 User Profile")
    user_input = st.sidebar.text_area(
        "Enter your skills (comma separated)",
        placeholder="python, sql, ml"
    )
    analyze = st.sidebar.button("🚀 Analyze Career")

    if analyze:
        if user_input.strip() == "":
            st.warning("Please enter your skills")
        else:
            user_input = user_input.lower()

            # -------- ROLE PREDICTION --------
            role_vectorizer = TfidfVectorizer()
            role_vectors = role_vectorizer.fit_transform(df["skills"].str.lower())
            user_vector = role_vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vector, role_vectors)[0]
            df["match_score"] = similarities
            top_roles = df.sort_values("match_score", ascending=False).head(3)

            # -------- DASHBOARD CARDS: TOP ROLES --------
            st.subheader("🎯 Top Matching Job Roles")
            cols = st.columns(3)
            for i, col in enumerate(cols):
                if i < len(top_roles):
                    col.markdown(f"""
                    <div style="background-color:#1e1e1e; padding:15px; border-radius:12px; text-align:center;">
                        <h4 style="color:#ffcc00;">{top_roles.iloc[i]['job_title']}</h4>
                        <p style="color:#00ffcc; font-size:16px;">Score: {round(top_roles.iloc[i]['match_score'],2)}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # -------- SKILL RECOMMENDATION --------
            user_skills = set([s.strip() for s in user_input.split(",")])
            all_skills = set()
            for skills in df["skills"]:
                for s in skills.split(","):
                    all_skills.add(s.strip().lower())
            missing_skills = list(all_skills - user_skills)

            skill_scores = {}
            for skill in missing_skills:
                combined_text = user_input + ", " + skill
                X_test = vectorizer.transform([combined_text])
                score = model.predict(X_test)[0]
                skill_scores[skill] = score

            recommended = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            # -------- DASHBOARD CARDS: TOP SKILLS --------
            st.subheader("🔥 Recommended Skills to Learn")
            skill_cards = pd.DataFrame(recommended, columns=['Skill','Score'])
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(skill_cards):
                    col.markdown(f"""
                    <div style="background-color:#1e1e1e; padding:10px; border-radius:12px; text-align:center;">
                        <h5 style="color:#00ffcc;">{skill_cards.loc[i,'Skill'].upper()}</h5>
                        <p style="color:#ffcc00;">Score: {round(skill_cards.loc[i,'Score'],2)}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # -------- INTERACTIVE SKILL TREND GRAPH --------
            st.subheader("📊 Skill Trend / Future Demand")
            fig = px.bar(
                skill_cards,
                x='Score',
                y='Skill',
                orientation='h',
                text='Score',
                color='Score',
                color_continuous_scale='teal'
            )
            fig.update_layout(
                template='plotly_dark',
                xaxis_title='Demand Score',
                yaxis_title='Skill',
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

            # -------- SAVE USER HISTORY --------
            history = {
                "username": username,
                "skills": user_input,
                "recommended_skills": ", ".join([s.upper() for s,_ in recommended]),
                "timestamp": datetime.datetime.now()
            }
            history_df = pd.DataFrame([history])
            history_file = "dataset/user_history.csv"

            if os.path.exists(history_file):
                history_df.to_csv(history_file, mode="a", header=False, index=False)
            else:
                history_df.to_csv(history_file, index=False)

            st.subheader("📜 Your Recent Submission")
            st.dataframe(history_df)

    # -------- SHOW USER HISTORY --------
    st.subheader("🗂️ Your Submission History")
    history_file = "dataset/user_history.csv"
    if os.path.exists(history_file):
        all_history = pd.read_csv(history_file)
        user_data = all_history[all_history['username'] == username]
        if not user_data.empty:
            st.dataframe(user_data)
        else:
            st.info("No previous submissions yet.")

    st.markdown('<div class="footer">Final Year ML Project | Built with Streamlit</div>', unsafe_allow_html=True)
