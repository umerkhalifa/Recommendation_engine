import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import time

class ScholarlinkRecommendationEngine:
    def __init__(self, student_embeddings_path, professor_embeddings_path, 
                 student_guids_path, professor_guids_path):
        try:
            with open(student_embeddings_path, 'rb') as f:
                self.student_embeddings = np.load(f)

            with open(professor_embeddings_path, 'rb') as f:
                self.professor_embeddings = np.load(f)

            with open(student_guids_path, 'r') as f:
                self.student_guids = json.load(f)

            with open(professor_guids_path, 'r') as f:
                self.professor_guids = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    def recommend_professors(self, student_id, top_n=10, threshold=0.90):
        try:
            student_index = self.student_guids.index(student_id)
            student_embedding = self.student_embeddings[student_index].reshape(1, -1)

            sim_scores = cosine_similarity(student_embedding, self.professor_embeddings)
            
            indices = np.argsort(sim_scores[0])[-top_n:][::-1]
            
            mask = sim_scores[0, indices] > threshold
            filtered_indices = indices[mask]
            
            recommended_professors = [(self.professor_guids[i], sim_scores[0, i]*100) for i in filtered_indices]
            
            return recommended_professors
        except ValueError as e:
            print(f"Error: {e}")
            return []

    def recommend_students(self, professor_id, top_n=10, threshold=0.90):
        try:
            professor_index = self.professor_guids.index(professor_id)
            professor_embedding = self.professor_embeddings[professor_index].reshape(1, -1)
            
            sim_scores = cosine_similarity(professor_embedding, self.student_embeddings)
            
            indices = np.argsort(sim_scores[0])[-top_n:][::-1]
            
            mask = sim_scores[0, indices] > threshold
            filtered_indices = indices[mask]
            
            recommended_students = [(self.student_guids[i], sim_scores[0, i]*100) for i in filtered_indices]
            
            return recommended_students
        except ValueError as e:
            print(f"Error: {e}")
            return []
        
    def recommend_students_to_students(self, student_id, top_n=10, threshold=0.90):
        try:
            student_index = self.student_guids.index(student_id)
            student_embedding = self.student_embeddings[student_index].reshape(1, -1)

            sim_scores = cosine_similarity(student_embedding, self.student_embeddings)
            
            indices = np.argsort(sim_scores[0])[-top_n:][::-1]
            
            mask = sim_scores[0, indices] > threshold
            filtered_indices = indices[mask]
            
            recommended_students = [(self.student_guids[i], sim_scores[0, i]*100) for i in filtered_indices]
            
            return recommended_students
        except ValueError as e:
            print(f"Error: {e}")
            return []

    def recommend_professors_to_professors(self, professor_id, top_n=10, threshold=0.90):
        try:
            professor_index = self.professor_guids.index(professor_id)
            professor_embedding = self.professor_embeddings[professor_index].reshape(1, -1)
            
            sim_scores = cosine_similarity(professor_embedding, self.professor_embeddings)
            
            indices = np.argsort(sim_scores[0])[-top_n:][::-1]
            
            mask = sim_scores[0, indices] > threshold
            filtered_indices = indices[mask]
            
            recommended_professors = [(self.professor_guids[i], sim_scores[0, i]*100) for i in filtered_indices]
            
            return recommended_professors
        except ValueError as e:
            print(f"Error: {e}")
            return []
        

if __name__ == "__main__":
    start = time.time()
    student_embeddings_path = './data/recommender_data/student_embeddings.npy'
    professor_embeddings_path = './data/recommender_data/professor_embeddings.npy'
    student_guids_path = './data/recommender_data/student_guids.json'
    professor_guids_path = './data/recommender_data/professor_guids.json'
    
    recommender = ScholarlinkRecommendationEngine(student_embeddings_path, professor_embeddings_path, 
                                              student_guids_path, professor_guids_path)
    
    student_id = '633a3646-53e5-4f15-9afa-d9d7fc959df8'
    recommended_professors = recommender.recommend_professors(student_id)
    print(f"Time elapsed for recommending professors for {student_id} is {time.time() - start} seconds")
    
    start = time.time()
    recommended_professors = recommender.recommend_students_to_students(student_id)
    print(f"Time elapsed for recommending professors for {student_id} is {time.time() - start} seconds")

    start = time.time()
    professor_id = '724a1946-8e63-459e-8975-b0c2b0f75567'
    recommended_students = recommender.recommend_students(professor_id)
    print(f"Time elapsed for recommending students for {professor_id} is {time.time() - start} seconds")

    start = time.time()
    recommended_students = recommender.recommend_professors_to_professors(professor_id)
    print(f"Time elapsed for recommending students for {professor_id} is {time.time() - start} seconds")