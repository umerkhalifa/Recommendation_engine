# Recommendation_engine
This project was built as part of an interview for the Senior Data Scientist Position (NLP) at Scholarlink.ai

# Requirements and installation
Versions of libraries are for Python 3.9.16 
install libraries by running the following command
"pip install -r requirements.txt"

# Steps to Execute the Project
  
1. python dataprep.py (Generates synthetic research Interests data using the unique combinations for each University Field)

2. python recommendation_dataprep.py (Generates GUIDS and embeddings for professor and students and the data is stored in 'data/recommender_data')

3. python recommendation_engine.py (Recommends Students and professor for the respective given GUID)

4. python evaluate.py (To evaluate the recommendation engine)

5. streamlit run app.py (To run the recommendation engine has app)
