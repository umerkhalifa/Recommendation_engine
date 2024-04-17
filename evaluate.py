import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from recommendation_dataprep import RecommenderDataPrep
from recommendation_engine import ScholarlinkRecommendationEngine

class DataPreparation:
    def __init__(self, students_path, professors_path):
        self.students = pd.read_csv(students_path)
        self.professors = pd.read_csv(professors_path)

    def clean_text(self, text):
        try:
            result = text.split(',')
            result = [item.strip() for item in result]
            result = list(set(sorted(result)))
            result = ','.join(result)
            return result
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return None

    def select_candidates(self, df):
        try:
            selected_df = pd.DataFrame()
            for field in df['University Field'].unique():
                selected_df = pd.concat([selected_df, df[df['University Field'] == field].head(100)], axis=0)
            return selected_df
        except Exception as e:
            print(f"Error selecting candidates: {e}")
            return None

    def candidates(self, df):
        try:
            selected_df = pd.DataFrame()
            for field in df['University Field'].unique():
                candidates = df[df['University Field'] == field].head(50)
                candidates['University Field'] = np.random.choice(df['University Field'].unique(), len(candidates))
                selected_df = pd.concat([selected_df, candidates], axis=0)
            return selected_df
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return None

    def prepare_test_data(self, students_output_path, professors_output_path):
        try:
            s = self.select_candidates(self.students)
            s = self.candidates(s)
            s['Research Interests'] = s['Research Interests'].apply(self.clean_text)

            p = self.select_candidates(self.professors)
            p = self.candidates(p)
            p['Research Interests'] = p['Research Interests'].apply(self.clean_text)

            s.to_csv(students_output_path, index=False)
            p.to_csv(professors_output_path, index=False)
        except Exception as e:
            print(f"Error preparing test data: {e}")

class EmbeddingCreation:
    def __init__(self, students_data_path, professors_data_path, data_download_path):
        self.students_data_path = students_data_path
        self.professors_data_path = professors_data_path
        self.data_download_path = data_download_path

    def create_embeddings(self, existing_student_embeddings=None, existing_professor_embeddings=None):
        try:
            data_prep = RecommenderDataPrep(self.students_data_path, self.professors_data_path, self.data_download_path)
            students, professors = data_prep.load_data()
            student_embeddings, professor_embeddings = data_prep.create_embeddings(students, professors, existing_student_embeddings, existing_professor_embeddings)
            
            student_guids = students['Student GUID'].tolist()
            professor_guids = professors['Professor GUID'].tolist()

            data_prep.save_embeddings_and_guids(student_embeddings, professor_embeddings, student_guids, professor_guids)
        except Exception as e:
            print(f"Error creating embeddings: {e}")


class RecommendationTest:
    def __init__(self, student_embeddings_path, professor_embeddings_path, 
                 student_guids_path, professor_guids_path, data_download_path):
        self.student_embeddings_path = student_embeddings_path
        self.professor_embeddings_path = professor_embeddings_path
        self.student_guids_path = student_guids_path
        self.professor_guids_path = professor_guids_path
        self.data_download_path = data_download_path

    def evaluate_recommendations(self, student_id, professor_id):
        try:
            recommender = ScholarlinkRecommendationEngine(self.student_embeddings_path, self.professor_embeddings_path, 
                                                          self.student_guids_path, self.professor_guids_path)

            # Professors to Student
            recommendations = recommender.recommend_professors(student_id)
            df = pd.DataFrame(recommendations, columns=['Professor GUID', 'Similarity Score'])
            df.to_csv(f"{self.data_download_path}/Professors_match_for_student_id({student_id}).csv", index=False)

            # Students to Student
            recommendations = recommender.recommend_students_to_students(student_id)
            df = pd.DataFrame(recommendations, columns=['Student GUID', 'Similarity Score'])
            df.to_csv(f"{self.data_download_path}/Student_match_for_student_id({student_id}).csv", index=False)

            # Students to Professor
            recommendations = recommender.recommend_students(professor_id)
            df = pd.DataFrame(recommendations, columns=['Student GUID', 'Similarity Score'])
            df.to_csv(f"{self.data_download_path}/Students_match_for_professor_id({professor_id}).csv", index=False)

            # Professors to Professor
            recommendations = recommender.recommend_professors_to_professors(professor_id)
            df = pd.DataFrame(recommendations, columns=['Professor GUID', 'Similarity Score'])
            df.to_csv(f"{self.data_download_path}/Professors_match_for_professor_id({professor_id}).csv", index=False)
        except Exception as e:
            print(f"Error evaluating recommendations: {e}")

if __name__ == "__main__":
    # Test data creation
    students_path = './data/raw/students.csv'
    professors_path = './data/raw/professors.csv'
    students_output_path = './data/evaluate_data/students_test.csv'
    professors_output_path = './data/evaluate_data/professors_test.csv'

    data_prep = DataPreparation(students_path, professors_path)
    data_prep.prepare_test_data(students_output_path, professors_output_path)

    # Creating test data embedding
    students_data_path = './data/evaluate_data/students_test.csv'
    professors_data_path = './data/evaluate_data/professors_test.csv'
    data_download_path = './data/evaluate_data'

    embedding_creation = EmbeddingCreation(students_data_path, professors_data_path, data_download_path)
    embedding_creation.create_embeddings()


    # Recommendation Engine Test
    student_embeddings_path = './data/evaluate_data/student_embeddings.npy'
    professor_embeddings_path = './data/evaluate_data/professor_embeddings.npy'
    student_guids_path = './data/evaluate_data/student_guids.json'
    professor_guids_path = './data/evaluate_data/professor_guids.json'
    data_download_path = './data/evaluate_data'

    recommendation_test = RecommendationTest(student_embeddings_path, professor_embeddings_path, 
                                            student_guids_path, professor_guids_path, data_download_path)

    student_id = '2e8bb391-c77b-4e79-8f67-fd6614f5335a'
    professor_id = '3ff367d3-dd69-461a-9d4f-e9c498a8859c'

    recommendation_test.evaluate_recommendations(student_id, professor_id)
