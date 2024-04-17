import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_engine import ScholarlinkRecommendationEngine

def main():
    st.title("Scholarlink Recommendation Engine")
    
    student_embeddings_path = './data/recommender_data/student_embeddings.npy'
    professor_embeddings_path = './data/recommender_data/professor_embeddings.npy'
    student_guids_path = './data/recommender_data/student_guids.json'
    professor_guids_path = './data/recommender_data/professor_guids.json'

    recommender = ScholarlinkRecommendationEngine(student_embeddings_path, professor_embeddings_path, 
                                                  student_guids_path, professor_guids_path)
    
    user_input = st.text_input("Enter Student or Professor ID:", "")
    
    option = st.selectbox('Select Recommendation Type:', 
                          ['Recommend Professors for Student', 
                           'Recommend Students for Student', 
                           'Recommend Students for Professor',
                           'Recommend Professors for Professor'])
    
    if user_input:
        start = time.time()
        
        if option == 'Recommend Professors for Student':
            recommended_items = recommender.recommend_professors(user_input)
            df = pd.DataFrame(recommended_items, columns=['Professor GUID', 'Similarity Score'])
            st.write(df)
            st.write(f"Time elapsed: {time.time() - start} seconds")
            
        elif option == 'Recommend Students for Student':
            recommended_items = recommender.recommend_students_to_students(user_input)
            df = pd.DataFrame(recommended_items, columns=['Student GUID', 'Similarity Score'])
            st.write(df)
            st.write(f"Time elapsed: {time.time() - start} seconds")
            
        elif option == 'Recommend Students for Professor':
            recommended_items = recommender.recommend_students(user_input)
            df = pd.DataFrame(recommended_items, columns=['Student GUID', 'Similarity Score'])
            st.write(df)
            st.write(f"Time elapsed: {time.time() - start} seconds")
            
        elif option == 'Recommend Professors for Professor':
            recommended_items = recommender.recommend_professors_to_professors(user_input)
            df = pd.DataFrame(recommended_items, columns=['Professor GUID', 'Similarity Score'])
            st.write(df)
            st.write(f"Time elapsed: {time.time() - start} seconds")


if __name__ == "__main__":
    main()
