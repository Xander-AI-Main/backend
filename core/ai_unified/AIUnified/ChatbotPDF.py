import fitz
import google.generativeai as genai
import os
import uuid

api_key = "AIzaSyC93XxpL8z7dz4UjNBvECFYaobAOQre0Bk"
genai.configure(api_key=api_key)

class ChatbotPDF:
    def __init__(self, dataset_url, userId, task):
        self.dataset_url = dataset_url
        self.pdf_dir = 'pdfs'
        self.pdf_name = self.dataset_url.split('/')[-1]
        self.pdf_path = os.path.join(self.pdf_dir, self.pdf_name)
        self.userId = userId
        self._id = str(uuid.uuid4())
        self.task = task

    # def extract(self):
    # self.pdf_name = self.dataset_url.split('/')[-1]
    #     self.pdf_path = os.path.join(self.pdf_dir, self.pdf_name)
    #     pdf_document = fitz.open(self.pdf_path)

    #     text = ""
    #     for page_num in range(pdf_document.page_count):
    #         page = pdf_document.load_page(page_num)
    #         text += page.get_text()

    #     pdf_document.close()

    #     return text

    def execute(self):
        # if not os.path.exists(self.pdf_path):
        #     raise FileNotFoundError(f"The file {self.pdf_name} does not exist in the directory {self.pdf_dir}")
        
        # text = self.extract()

        # model = genai.GenerativeModel("gemini-1.5-flash")
        # response = model.generate_content(f"Context: {text} Answer the following question in less than 100 words no matter what and if the answer doesnt exist in the context, simple reply with answer not available: {self.question}")

        # print(response.text)
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
            "modelUrl": "",
            "helpers": [],
            "id": self._id,
            "architecture": "Language Models",
            "hyperparameters": {},
            "size": 0,
            "interferenceCode": interference_code,
            "datasetUrl": self.dataset_url,
            "codes": [
                    {"python": api_code_python},
                    {"javascript": api_code_js}
            ],
            "task": self.task,
        }

        return model_obj


