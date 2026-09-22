# AI JOB RECOMMENDATION SYSTEM 


AI-powered job recommendation engine that matches resumes/skills to job postings using TF-IDF vectorization and cosine similarity. Includes a Streamlit web app, CLI demo, and unit tests. Fully explainable content-based filtering no black-box model.




## About this project 

An AI-powered job recommendation engine that matches a candidate's resume or skills against open job postings and returns ranked, explainable recommendations — built to demonstrate a real, working recommender system end to end.

Paste in a resume or a list of skills, and the system scores every job posting by textual similarity, ranks the results, and shows exactly which skills matched and which are missing — turning a black-box "AI match" into something a candidate (or an interviewer) can actually verify.

## How it works

This is a content-based filtering recommender, the same family of technique used across many real-world recommendation engines:

- TF-IDF Vectorization — every job posting is converted into a numeric vector using Term Frequency–Inverse Document Frequency, which automatically down-weights generic words and up-weights distinctive, skill-specific terms.
- Cosine Similarity — the candidate's resume/skills text is vectorized with the same vocabulary and compared against every job vector, producing a 0–1 match score.
- Ranking & Filtering — results are sorted by score and can be filtered by location, experience level, or minimum match threshold.
  
No neural network, no external API calls, no training data required — it's fast, fully local, and every recommendation can be explained in plain English (which is exactly what the matched/missing skill breakdown does).


## Tech stack used

Python · pandas · scikit-learn (TfidfVectorizer, cosine_similarity) · Streamlit · pypdf




## Demo Link
https://ai-job-recommendation-system-s.streamlit.app/   
