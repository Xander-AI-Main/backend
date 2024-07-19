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

model_zip_url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/sentence_transformer_model.zip'

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

question_embeddings_url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/question_embeddings.pt'
answer_embeddings_url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/answer_embeddings.pt'

response = requests.get(question_embeddings_url)
with open('question_embeddings.pt', 'wb') as file:
    file.write(response.content)

response = requests.get(answer_embeddings_url)
with open('answer_embeddings.pt', 'wb') as file:
    file.write(response.content)

question_embeddings = torch.load('question_embeddings.pt')
answer_embeddings = torch.load('answer_embeddings.pt')

nltk.download('punkt')
nltk.download('stopwords')

data_url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/data_new.json'

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

print(f"Answer: {answer}")
print(f"Similarity Percentage: {similarity_percentage:.2f}%")