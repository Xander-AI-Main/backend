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
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.api_url = 'https://apiv3.xanderco.in/core/store/'
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        self.json_url = dataset_url
        self.qa_data = self.fetch_json_data(self.json_url)
        self.questions = [item['question'] for item in self.qa_data]
        self.answers = [item['answer'] for item in self.qa_data]
        self.directory_path = "models"
        self.que_path = f"question_embeddings{str(uuid.uuid4())}.pt"
        self.ans_path = f'answer_embeddings{str(uuid.uuid4())}.pt'
        self.que_complete_path = os.path.join(self.directory_path, self.que_path)
        self.ans_complete_path = os.path.join(self.directory_path, self.ans_path)
        self.userId = userId

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

        files_to_upload = [self.que_complete_path, self.ans_complete_path]

        for file_path in files_to_upload:
            file = {
                'file': open(file_path, 'rb')
            }

            try:
                response = requests.post(self.api_url, files=file)
                response_data = response.json()
                print(response_data)
                if response.status_code == 201 or response.status_code == 200:
                    pdf_info = response_data.get('file_url')
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

        torch.save(question_embeddings,  self.que_complete_path)
        torch.save(answer_embeddings, self.ans_complete_path)

        model_path = "https://xanderco-storage.s3.ap-south-1.amazonaws.com/sentence_transformer_model.zip"

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

question_embeddings_url = '{uploaded_urls.get(self.que_complete_path, "")}'
answer_embeddings_url =  '{uploaded_urls.get(self.ans_complete_path, "")}'

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

        api_code_python = f'''
import requests
import json

url = "https://apiv3.xanderco.in/core/interference/" 

data = {{
    "data": "Your input text",
    "modelId": '{_id}',
    "userId": '{self.userId}',
}}

try:
    response = requests.post(url, json=data)

    if response.status_code == 200:
        # Print the response JSON
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {{response.status_code}}")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {{e}}")
'''

        api_code_js = f'''
const url = "https://apiv3.xanderco.in/core/interference/";

const data = {{
    data: "Your text here",
    modelId: '{_id}',
    userId: '{self.userId}',
}};

const headers = {{
    'Content-Type': 'application/json',
}};

fetch(url, {{
    method: 'POST',
    headers: headers,
    body: JSON.stringify(data)
}})
.then(response => response.json().then(data => {{
    if (response.ok) {{
        console.log("Response:");
        console.log(JSON.stringify(data, null, 2));
    }} else {{
        console.error(`Error: ${{response.status}}`);
        console.error(data);
    }}
}}))
.catch(error => {{
    console.error(`An error occurred: ${{error}}`);
}});
'''
        
        model_obj = {
            "modelUrl": model_path,
            "helpers": [{"question_embeddings": uploaded_urls.get(self.que_complete_path, "")}, {"answer_embeddings": uploaded_urls.get(self.ans_complete_path, "")}],
            "id": _id,
            "architecture": "Sentence Transformers",
            "hyperparameters": {},
            "size": os.path.getsize(self.que_complete_path) / (1024 ** 3) + os.path.getsize(self.ans_complete_path) / (1024 ** 3),
            "task": self.task,
            "interferenceCode": interference_code,
            "datasetUrl": self.dataset_url,
                "codes": [
                    {"python": api_code_python},
                    {"javascript": api_code_js}
                ]
        }

        # os.remove(self.que_path)
        # os.remove(self.ans_path)
        return model_obj
