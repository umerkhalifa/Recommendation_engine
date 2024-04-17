import pandas as pd
from faker import Faker
import numpy as np
import random
from sentence_transformers import SentenceTransformer

class UniversityDataGenerator:
    def __init__(self, data_path, download_path, model_name):
        self.data_path = data_path
        self.download_path = download_path
        self.fake = Faker()
        self.model = SentenceTransformer(model_name)
        
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

    def generate_research_interests(self, university_field):
        try:
            if university_field == 'Physics':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Astrophysics', 'Condensed Matter Physics', 'Cosmology', 'Experimental Physics', 'Nuclear Physics', 'Particle Physics', 'Photonics', 'Quantum Mechanics', 'Statistical Mechanics', 'Theoretical Physics'])
            elif university_field == 'Psychology':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Biopsychology', 'Clinical Psychology', 'Cognitive Psychology', 'Developmental Psychology', 'Experimental Psychology', 'Forensic Psychology', 'Health Psychology', 'Personality Psychology', 'Psychometrics', 'Social Psychology'])
            elif university_field == 'Chemistry':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Analytical Chemistry', 'Biochemistry', 'Environmental Chemistry', 'Inorganic Chemistry', 'Materials Science', 'Medicinal Chemistry', 'Organic Chemistry', 'Physical Chemistry', 'Polymer Science', 'Theoretical Chemistry'])
            elif university_field == 'History':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Ancient History', 'Cultural History', 'Economic History', 'History of Science', 'Medieval History', 'Military History', 'Modern History', 'Political History', 'Social History', 'World History'])
            elif university_field == 'Mathematics':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Algebra', 'Applied Mathematics', 'Calculus', 'Differential Equations', 'Geometry', 'Mathematical Physics', 'Number Theory', 'Probability', 'Statistics', 'Topology'])
            elif university_field == 'Literature':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Comparative Literature', 'Creative Writing', 'Drama', 'Historiography', 'Literary Criticism', 'Narrative Theory', 'Novel', 'Poetry', 'Rhetoric', 'World Literature'])
            elif university_field == 'Engineering':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Aerospace Engineering', 'Biomedical Engineering', 'Chemical Engineering', 'Civil Engineering', 'Electrical Engineering', 'Environmental Engineering', 'Industrial Engineering', 'Mechanical Engineering', 'Software Engineering', 'Systems Engineering'])
            elif university_field == 'Computer Science':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Artificial Intelligence', 'Blockchain', 'Cloud Computing', 'Computer Vision', 'Cybersecurity', 'Data Science', 'Human-Computer Interaction', 'Machine Learning', 'Quantum Computing', 'Software Engineering'])
            elif university_field == 'Biology':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Biochemistry', 'Cell Biology', 'Conservation Biology', 'Ecology', 'Evolutionary Biology', 'Genetics', 'Immunology', 'Marine Biology', 'Microbiology', 'Neuroscience'])
            elif university_field == 'Economics':
                return self.fake.words(nb=random.randint(3, 3), ext_word_list=['Behavioral Economics', 'Development Economics', 'Econometrics', 'Financial Economics', 'Health Economics', 'International Economics', 'Labor Economics', 'Macroeconomics', 'Microeconomics', 'Public Economics'])
            else:
                return None
        except Exception as e:
            print(f"Error generating research interests: {e}")
            return None

    def generate_data(self):
        try:
            students = pd.read_excel(self.data_path, sheet_name='Students')
            professors = pd.read_excel(self.data_path, sheet_name='Professors')

            students['Research Interests'] = students.apply(lambda x: self.clean_text(",".join(self.generate_research_interests(x['University Field']))), axis=1)
            professors['Research Interests'] = professors.apply(lambda x: self.clean_text(",".join(self.generate_research_interests(x['University Field']))), axis=1)

            students.to_csv(f'{self.download_path}/students.csv', index=False)
            professors.to_csv(f'{self.download_path}/professors.csv', index=False)
            
            print("Data generation completed successfully.")
        except Exception as e:
            print(f"Error generating data: {e}")

if __name__ == "__main__":
    data_path = './data/raw/university_data.xlsx'
    download_path = './data/raw'
    model_name = 'paraphrase-MiniLM-L6-v2'
    data_generator = UniversityDataGenerator(data_path, download_path,  model_name)
    data_generator.generate_data()
