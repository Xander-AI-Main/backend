import requests
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util
import shutil
import torch
import zipfile
import uuid
import os

class Chatbot:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.api_url = 'https://s3-api-uat.idesign.market/api/upload'
        self.bucket_name = 'idesign-quotation'
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        self.json_url = dataset_url
        self.qa_data = self.fetch_json_data(self.json_url)
        self.questions = [item['question'] for item in self.qa_data]
        self.answers = [item['answer'] for item in self.qa_data]

    def fetch_json_data(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status() 
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching JSON data: {str(e)}")
            return None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  
        text = re.sub(r'\s+', ' ', text).strip()  
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def encode_embeddings(self):
        processed_questions = [self.preprocess_text(question) for question in self.questions]
        processed_answers = [self.preprocess_text(answer) for answer in self.answers]

        question_embeddings = self.model.encode(processed_questions, convert_to_tensor=True)
        answer_embeddings = self.model.encode(processed_answers, convert_to_tensor=True)

        return question_embeddings, answer_embeddings

    def upload_files_to_s3(self):
        uploaded_urls = {}

        files_to_upload = ['sentence_transformer_model.zip', 'question_embeddings.pt', 'answer_embeddings.pt']

        for file_path in files_to_upload:
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(file_path, 'rb')
            }
            
            try:
                response = requests.put(self.api_url, files=files)
                response_data = response.json()
                
                if response.status_code == 200:
                    pdf_info = response_data.get('locations', [])[0]
                    initial_url = pdf_info
                    uploaded_urls[file_path] = initial_url
                    print(f"File {file_path} uploaded successfully.")
                else:
                    print(f"Failed to upload file {file_path}. Error: {response_data.get('error')}")
            
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {str(e)}")
        
        return uploaded_urls
    
    def execute(self):
        self.qa_data = self.fetch_json_data(self.dataset_url)
        
        self.questions = [item['question'] for item in self.qa_data]
        self.answers = [item['answer'] for item in self.qa_data]

        question_embeddings, answer_embeddings = self.encode_embeddings()

        torch.save(question_embeddings, 'question_embeddings.pt')
        torch.save(answer_embeddings, 'answer_embeddings.pt')
        
        model_path = 'sentence_transformer_model.zip'
        with zipfile.ZipFile(model_path, 'w') as model_zip:
            for file in ['question_embeddings.pt', 'answer_embeddings.pt']:
                model_zip.write(file)

        uploaded_urls = self.upload_files_to_s3()

        _id = str(uuid.uuid4())
        model_obj = {
            "question_embeddings_url": uploaded_urls.get('question_embeddings.pt', ""),
            "answer_embeddings_url": uploaded_urls.get('answer_embeddings.pt', ""),
            "model_url": uploaded_urls.get(model_path, ""),
            "id": _id,
            "architecture": "Sentence Transformers",
            "hyperparameters": {},
            "size": os.path.getsize(model_path) / (1024 ** 3) + os.path.getsize("question_embeddings.pt") / (1024 ** 3) + os.path.getsize("answer_embeddings.pt") / (1024 ** 3)
        }

        return model_obj
