# Requirements and installation
Versions of libraries are for Python 3.9.16 
install libraries by running the following command
"pip install -r requirements.txt"

# Steps to Execute the project
  Generates synthetic research Interests data using the unique combinations for each University Field
1. python dataprep.py

  Generates GUIDS and embeddings for professor and students and the data is stored in 'data/recommender_data'
2. python faiss_dataprep.py

  Recommends Students and professor for the respective given GUID
3. python faiss_approach.py
