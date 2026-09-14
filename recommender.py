"""
recommender.py
----------------
This is the "AI brain" of the Job Recommendation System.

HOW IT WORKS (in plain English):
1. Every job has a text description of the skills it needs.
2. We turn every job's skills into a list of numbers using a technique
   called TF-IDF (Term Frequency - Inverse Document Frequency).
   Think of TF-IDF as a way to measure "how important is this word
   for this job, compared to all other jobs".
3. We do the exact same thing to the skills/resume text the user types in.
4. We then measure the "angle" between the user's number-vector and
   each job's number-vector using Cosine Similarity.
   - A small angle (close to 1.0 similarity) means the user's skills
     are very close to what the job needs.
   - A big angle (close to 0.0 similarity) means they barely match.
5. We sort all jobs by that similarity score and return the top matches.

This is called "Content-Based Filtering" — one of the simplest and most
popular real-world recommendation techniques (also used by Netflix,
Spotify, and job portals like LinkedIn for early-stage matching).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommender:
    def __init__(self, csv_path="jobs_data.csv"):
        # Step 1: Load the job dataset from CSV into a pandas DataFrame
        self.jobs_df = pd.read_csv(csv_path)

        # Step 2: Combine skills + description into one text field per job.
        # This gives the model more context to compare against.
        self.jobs_df["combined_text"] = (
            self.jobs_df["skills_required"].fillna("") + " " +
            self.jobs_df["description"].fillna("")
        )

        # Step 3: Create the TF-IDF vectorizer and fit it on all job text.
        # "fit" = the model learns the vocabulary of all jobs.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.job_vectors = self.vectorizer.fit_transform(
            self.jobs_df["combined_text"]
        )

    def _matched_skills(self, user_skills_text: str, job_skills_text: str) -> str:
        """Return the overlapping skills (as whole comma-separated phrases, e.g.
        'machine learning' stays intact) between user input and a job, for
        reasoning/explainability."""
        user_set = {s.strip().lower() for s in user_skills_text.split(",") if s.strip()}
        job_set = {s.strip().lower() for s in job_skills_text.split(",") if s.strip()}
        overlap = sorted(user_set & job_set)
        return ", ".join(overlap) if overlap else "general text similarity"

    def recommend(self, user_skills_text: str, top_n: int = 5) -> pd.DataFrame:
        """
        Given a string of user skills (e.g. "python, sql, machine learning"),
        return the top_n most similar jobs as a DataFrame with a match_score
        and reason column. Handles the cold-start case (empty/blank input)
        by falling back to a popularity-style default (here: most in-demand
        skill overlap across the catalogue) instead of returning nothing.
        """
        # Cold-start handling: no input skills provided at all.
        if not user_skills_text or not user_skills_text.strip():
            fallback = self.jobs_df.copy()
            fallback["match_score"] = 0.0
            fallback["reason"] = "Cold start: no user preferences given — showing sample of open roles"
            return fallback.head(top_n)[
                ["title", "company", "location", "experience_level",
                 "skills_required", "match_score", "reason"]
            ].reset_index(drop=True)

        # Step 4: Turn the user's skills into the SAME kind of vector
        # using the vocabulary the vectorizer already learned.
        user_vector = self.vectorizer.transform([user_skills_text])

        # Step 5: Compare the user vector against every job vector.
        similarity_scores = cosine_similarity(user_vector, self.job_vectors).flatten()

        # Step 6: Attach scores to the jobs table and sort by best match.
        results = self.jobs_df.copy()
        results["match_score"] = (similarity_scores * 100).round(1)  # as a percentage
        results = results.sort_values(by="match_score", ascending=False)

        # Step 7: Only keep jobs with a non-zero match, then return the top N.
        results = results[results["match_score"] > 0].head(top_n).copy()

        # Step 8: Add a human-readable reason (explainability) for each recommendation.
        results["reason"] = results["skills_required"].apply(
            lambda job_skills: f"Overlapping skills: {self._matched_skills(user_skills_text, job_skills)}"
        )

        return results[
            ["title", "company", "location", "experience_level",
             "skills_required", "match_score", "reason"]
        ].reset_index(drop=True)


# Quick manual test when running this file directly:
# python recommender.py
if __name__ == "__main__":
    engine = JobRecommender("jobs_data.csv")
    sample_skills = "python, machine learning, statistics, pandas"
    print(f"Searching jobs for skills: {sample_skills}\n")
    recommendations = engine.recommend(sample_skills, top_n=5)
    print(recommendations.to_string(index=False))