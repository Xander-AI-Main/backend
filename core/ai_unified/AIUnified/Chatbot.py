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
        print(self.json_url)
        self.qa_data = self.fetch_json_data(self.json_url)
        print(self.qa_data)
        self.questions = [item['question'] for item in self.qa_data]
        self.answers = [item['answer'] for item in self.qa_data]
        self.que_path = f"question_embeddings{str(uuid.uuid4())}.pt"
        self.ans_path = f'answer_embeddings{str(uuid.uuid4())}.pt'

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
        processed_questions = [self.preprocess_text(
            question) for question in self.questions]
        processed_answers = [self.preprocess_text(
            answer) for answer in self.answers]

        question_embeddings = self.model.encode(
            processed_questions, convert_to_tensor=True)
        answer_embeddings = self.model.encode(
            processed_answers, convert_to_tensor=True)

        return question_embeddings, answer_embeddings

    def upload_files_to_s3(self):
        uploaded_urls = {}

        files_to_upload = [self.que_path, self.ans_path]

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
                    print(
                        f"Failed to upload file {file_path}. Error: {response_data.get('error')}")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {str(e)}")

        return uploaded_urls

    def execute(self):
        self.qa_data = self.fetch_json_data(self.dataset_url)

        self.questions = [item['question'] for item in self.qa_data]
        self.answers = [item['answer'] for item in self.qa_data]

        question_embeddings, answer_embeddings = self.encode_embeddings()

        torch.save(question_embeddings,  self.que_path)
        torch.save(answer_embeddings, self.ans_path)

        model_path = "https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/sentence_transformer_model.zip"

        uploaded_urls = self.upload_files_to_s3()

        _id = str(uuid.uuid4())
        interference_code = f''' 
import requests
import torch
import zipfile
from sentence_transformers import SentenceTransformer
import os
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import util
import requests
import json

model_zip_url = '{model_path}'

extract_folder = './sentence_transformer_model'

os.makedirs(extract_folder, exist_ok=True)

print("Downloading model zip file...")
response = requests.get(model_zip_url, stream=True)
zip_file_path = './sentence_transformer_model.zip'

with open(zip_file_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

print("Unzipping model...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

model_path = os.path.join('sentence_transformer_model')
model = SentenceTransformer(model_path)

question_embeddings_url = '{uploaded_urls.get(self.que_path, "")}'
answer_embeddings_url =  '{uploaded_urls.get(self.ans_path, "")}'

response = requests.get(question_embeddings_url)
with open(question_embeddings_url.split("/")[-1], 'wb') as file:
    file.write(response.content)

response = requests.get(answer_embeddings_url)
with open(answer_embeddings_url.split("/")[-1], 'wb') as file:
    file.write(response.content)

question_embeddings = torch.load(question_embeddings_url.split("/")[-1])
answer_embeddings = torch.load(answer_embeddings_url.split("/")[-1])

nltk.download('punkt')
nltk.download('stopwords')

data_url = '{self.dataset_url}'

response = requests.get(data_url)
qa_data = json.loads(response.text)
questions = [item['question'] for item in qa_data]
answers = [item['answer'] for item in qa_data]

questions = [item['question'] for item in qa_data]
answers = [item['answer'] for item in qa_data]

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_answer(question):
    processed_question = preprocess_text(question)
    question_embedding = model.encode(processed_question, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]
    top_results = similarities.topk(k=5)
    print(top_results)
    similarity, index = similarities.max(), similarities.argmax()
    similarity_percentage = similarity.item() * 100
    
    if similarity_percentage > 45:
        return answers[index], similarity_percentage
    else: 
        return "Sorry, I didn't understand that!", similarity_percentage

user_question = "What is munafa"
answer, similarity_percentage = get_answer(user_question)

print(f"Answer: {{answer}}")
print(f"Similarity Percentage: {{similarity_percentage:.2f}}%")

        '''
        model_obj = {
            "modelUrl": "https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/sentence_transformer_model.zip",
            "helpers": [{"question_embeddings": uploaded_urls.get(self.que_path, "")}, {"answer_embeddings": uploaded_urls.get(self.ans_path, "")}],
            "id": _id,
            "architecture": "Sentence Transformers",
            "hyperparameters": {},
            "size": os.path.getsize(self.que_path) / (1024 ** 3) + os.path.getsize(self.ans_path) / (1024 ** 3),
            "task": self.task,
            "interferenceCode": interference_code,
            "datasetUrl": self.dataset_url
        }

        os.remove(self.que_path)
        os.remove(self.ans_path)

        return model_obj
