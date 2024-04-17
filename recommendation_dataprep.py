import os
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer
import pandas as pd

class RecommenderDataPrep:
    def __init__(self, students_data_path, professors_data_path, data_download_path, model_name='paraphrase-MiniLM-L6-v2'):
        self.students_data_path = students_data_path
        self.professors_data_path = professors_data_path
        self.data_download_path = data_download_path
        self.model = SentenceTransformer(model_name)
        self.last_timestamp = None

    def get_file_timestamp(self, path):
        try:
            return os.path.getmtime(path)
        except Exception as e:
            print(f"Error getting timestamp for {path}: {e}")
            return None

    def is_updated(self):
        current_timestamp = self.get_file_timestamp(self.students_data_path) + self.get_file_timestamp(self.professors_data_path)
        if current_timestamp != self.last_timestamp:
            self.last_timestamp = current_timestamp
            return True
        return False

    def clean_text(self, text):
        result = text.split(',')
        result = [item.strip() for item in result]
        result = ",".join(list(set(sorted(result))))
        return result

    def load_data(self):
        try:
            students = pd.read_csv(self.students_data_path)
            professors = pd.read_csv(self.professors_data_path)
            professors['Research Interests'] = professors['Research Interests'].apply(self.clean_text)
            students['Research Interests'] = students['Research Interests'].apply(self.clean_text)
            return students, professors
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def create_embeddings(self, students, professors, existing_student_embeddings=None, existing_professor_embeddings=None):
        try:
            student_embeddings = self.model.encode(students['Research Interests'].tolist())
            professor_embeddings = self.model.encode(professors['Research Interests'].tolist())
            
            if existing_student_embeddings is not None:
                student_embeddings = np.concatenate([existing_student_embeddings, student_embeddings], axis=0)
            
            if existing_professor_embeddings is not None:
                professor_embeddings = np.concatenate([existing_professor_embeddings, professor_embeddings], axis=0)
            
            return student_embeddings, professor_embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None, None

    def save_embeddings_and_guids(self, student_embeddings, professor_embeddings, student_guids, professor_guids):
        try:
            with open(f'{self.data_download_path}/student_embeddings.npy', 'wb') as f:
                np.save(f, student_embeddings)

            with open(f'{self.data_download_path}/professor_embeddings.npy', 'wb') as f:
                np.save(f, professor_embeddings)

            with open(f'{self.data_download_path}/student_guids.json', 'w') as f:
                json.dump(student_guids, f)

            with open(f'{self.data_download_path}/professor_guids.json', 'w') as f:
                json.dump(professor_guids, f)
        except Exception as e:
            print(f"Error saving embeddings and GUIDs: {e}")


if __name__ == "__main__":
    students_data_path = './data/raw/students.csv'
    professors_data_path = './data/raw/professors.csv'
    data_download_path = './data/recommender_data'

    data_prep = RecommenderDataPrep(students_data_path, professors_data_path, data_download_path)

    while True:
        if data_prep.is_updated():
            print("Dataset updated. Reloading data and updating embeddings...")
            
            students, professors = data_prep.load_data()

            if students is not None and professors is not None:
                print("Data loaded successfully.")

                existing_student_embeddings = None
                existing_professor_embeddings = None
                
                if os.path.exists(f'{data_download_path}/student_embeddings.npy'):
                    existing_student_embeddings = np.load(f'{data_download_path}/student_embeddings.npy')
                
                if os.path.exists(f'{data_download_path}/professor_embeddings.npy'):
                    existing_professor_embeddings = np.load(f'{data_download_path}/professor_embeddings.npy')

                student_embeddings, professor_embeddings = data_prep.create_embeddings(students, professors, existing_student_embeddings, existing_professor_embeddings)

                if student_embeddings is not None and professor_embeddings is not None:
                    print("Embeddings updated successfully.")

                    student_guids = students['Student GUID'].tolist()
                    professor_guids = professors['Professor GUID'].tolist()

                    data_prep.save_embeddings_and_guids(student_embeddings, professor_embeddings, student_guids, professor_guids)

                    print("Updated embeddings and GUIDs saved successfully.")
                else:
                    print("Failed to update embeddings.")
            else:
                print("Failed to load data.")
        
        time.sleep(5)  
