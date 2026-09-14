"""
app.py
-------
This is the web app (the part you SEE and click on).
It's built with Streamlit, a tool that turns a plain Python script
into a working website with almost no extra code — perfect for demos.

Run it with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from recommender import JobRecommender

# ---------- Page setup ----------
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="centered",
)

st.title("💼 AI-Powered Job Recommendation System")
st.write(
    "Paste your **skills** or a short summary of your **resume** below, "
    "and the AI will match you to the most relevant jobs from our database."
)

# ---------- Load the recommender engine (cached so it only loads once) ----------
@st.cache_resource
def load_engine():
    return JobRecommender("jobs_data.csv")

engine = load_engine()

# ---------- Sidebar: how it works + filters ----------
with st.sidebar:
    st.header("⚙️ How it works")
    st.write(
        "1. Your skills are converted into a numeric vector using **TF-IDF**.\n"
        "2. Each job's required skills are converted the same way.\n"
        "3. We compute **Cosine Similarity** between you and every job.\n"
        "4. The jobs with the highest similarity score are shown first."
    )
    st.divider()
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    exp_filter = st.multiselect(
        "Filter by experience level (optional)",
        options=sorted(engine.jobs_df["experience_level"].unique()),
    )

# ---------- Main input ----------
user_input = st.text_area(
    "Your skills (comma-separated) or resume summary",
    placeholder="e.g. python, sql, machine learning, data visualization",
    height=120,
)

search_clicked = st.button("🔍 Find Matching Jobs", type="primary")

# ---------- Run recommendation ----------
if search_clicked:
    if not user_input.strip():
        st.warning("Please enter at least a few skills first.")
    else:
        with st.spinner("Matching your skills to jobs..."):
            results = engine.recommend(user_input, top_n=top_n * 3)  # get extra, then filter

        if exp_filter:
            results = results[results["experience_level"].isin(exp_filter)]

        results = results.head(top_n)

        if results.empty:
            st.error("No matching jobs found. Try different or broader skills.")
        else:
            st.success(f"Found {len(results)} matching job(s)!")
            for _, row in results.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(row["title"])
                        st.write(f"**{row['company']}** · {row['location']} · {row['experience_level']}")
                        st.caption(f"Required skills: {row['skills_required']}")
                    with col2:
                        st.metric("Match Score", f"{row['match_score']}%")
                    st.progress(min(int(row["match_score"]), 100))

# ---------- Footer: show full dataset (nice for demo transparency) ----------
with st.expander("📊 View all jobs in the dataset"):
    st.dataframe(engine.jobs_df[["title", "company", "location", "experience_level", "skills_required"]])
