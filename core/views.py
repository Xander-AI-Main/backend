import os
from asgiref.sync import sync_to_async
import zipfile
import pandas as pd
import json
import mimetypes
import numpy as np
from django.conf import settings
from rest_framework import viewsets
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import DatasetUploadSerializer, DatasetSerializer, signupSerializer, TaskSerializer, ResultSerializer, InterferenceSerializer, FileUploadSerializer, ChatTrainSerializer
from .models import Dataset, userSignup
import requests
import json
import pandas as pd
from .ai_unified.AIUnified.RegressionML import RegressionML
from .ai_unified.AIUnified.RegressionDL import RegressionDL
from .ai_unified.AIUnified.ClassificationDL import ClassificationDL
from .ai_unified.AIUnified.ClassificationML import ClassificationML
from .ai_unified.AIUnified.ImageModelTrainer import ImageModelTrainer
from .ai_unified.AIUnified.TextModel import TextModel
from .ai_unified.AIUnified.Chatbot import Chatbot
from .ai_unified.AIUnified.ChatbotPDF import ChatbotPDF
from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import userSignup
from .serializers import signupSerializer, LoginSerializer, UserUpdateSerializer
from rest_framework.authtoken.models import Token
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from datetime import datetime, timedelta
from rest_framework.exceptions import ValidationError, NotFound, PermissionDenied
import time
import socket
import threading
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import uuid
import random
import numpy as np
import tensorflow as tf
import joblib
import requests
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import util
import torch
import chardet
from django.utils import timezone
import boto3
from botocore.config import Config
import fitz
import google.generativeai as genai
import kaggle
import shutil

api_key = "AIzaSyC93XxpL8z7dz4UjNBvECFYaobAOQre0Bk"
genai.configure(api_key=api_key)
geminiModel = genai.GenerativeModel("gemini-1.5-flash")
geminiProModel = genai.GenerativeModel("gemini-pro")

def remove_whitespace(filename):
    return ''.join(filename.split())

def checkTrain(dirs):
    for i, j in enumerate(dirs):
        if j.lower().find('train') != -1:
            if j.lower().endswith('.csv'):
                return {"index": i, "isDir": False}
            else:
                return {"index": i, "isDir": True}
    return {"index": -1, "isDir": False}


def checkTest(dirs):
    for i, j in enumerate(dirs):
        if j.lower().find('test') != -1:
            if j.lower().endswith('.csv'):
                return {"index": i, "isDir": False}
            else:
                return {"index": i, "isDir": True}
    return {"index": -1, "isDir": False}


def returnArch(data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]


url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/arch_new.json'

try:
    response = requests.get(url)

    if response.status_code == 200:
        arch_data = json.loads(response.content)
    else:
        print(
            f"Failed to retrieve data from {url}. Status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from {url}: {e}")


class signupViewset(viewsets.ModelViewSet):
    queryset = userSignup.objects.all()
    serializer_class = signupSerializer

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            response_serializer = self.get_serializer(user)

            userId = str(user.id)
            return Response({
                "message": "User created successfully",
                "userId": userId,
                "user": response_serializer.data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def returnArch(data, task, mainType, archType):
    current_task = arch_data[task]
    for i in current_task:
        if i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]


class LoginView(APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            token, created = Token.objects.get_or_create(user=user)

            user_serializer = signupSerializer(user)
            user_data = user_serializer.data

            user_data.pop('password', None)

            return Response({
                'token': token.key,
                'userId': str(user.id),
                'user': user_data
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download", url, ":", response.status_code)


def load_model_from_local(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print("Error loading model from path: " + str(e))
        return None


def load_scaler_from_local(path):
    try:
        scaler = joblib.load(path)
        return scaler
    except Exception as e:
        print("Error loading scaler from path: " + str(e))
        return None


def load_label_encoders_from_local(path):
    try:
        label_encoder = joblib.load(path)
        return label_encoder
    except Exception as e:
        print("Error loading label encoders from path: " + str(e))
        return None


def load_tokenizer(path):
    try:
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


def load_label_encoder(path):
    try:
        with open(path, 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return None


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def get_answer(question, model, question_embeddings, answer_embeddings, answers):
    processed_question = preprocess_text(question)
    question_embedding = model.encode(
        processed_question, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(
        question_embedding, question_embeddings)[0]
    similarity, index = similarities.max(), similarities.argmax()
    similarity_percentage = similarity.item() * 100

    if similarity_percentage > 45:
        return answers[index], similarity_percentage
    else:
        return "Sorry, I didn't understand that!", similarity_percentage


def numToText(finalColumn, x):
    arr = finalColumn.unique()
    return arr[x]


class Interference(APIView):
    def post(self, request):
        user_id = request.data.get('userId')
        model_id = request.data.get('modelId')
        input_data = request.data.get('data')
        image_file = request.FILES.get('image')
        image_url = request.data.get('imageUrl')

        user = get_object_or_404(userSignup, id=user_id)

        return_serializer = signupSerializer(user)
        data = return_serializer.data
        print(data["trained_model_url"])
        current_model = next(
            (model for model in data["trained_model_url"] if model["id"] == model_id), None)
        print(current_model)
        task = current_model["task"]
        datasetUrl = current_model["datasetUrl"]
        modelUrl = current_model["modelUrl"]
        predictions = None

        if task == "regression":
            helpers = current_model["helpers"]
            scalerUrl = helpers[0]["scaler"]
            labelUrl = helpers[1]["label_encoder"]

            model_name = modelUrl.split('/')[-1]
            scaler_name = scalerUrl.split('/')[-1]
            label_name = labelUrl.split('/')[-1]

            model_path = os.path.join("models", model_name)
            scaler_path = os.path.join("models", scaler_name)
            label_path = os.path.join("models", label_name)

            model = load_model_from_local(model_path)
            scaler = load_scaler_from_local(scaler_path)
            labelEncoder = load_label_encoders_from_local(label_path)

            if model and scaler:
                def preprocess_input(data, scaler, labelEncoder, categorical_columns, column_names):
                    df = pd.DataFrame([data], columns=column_names)
                    print(column_names)
                    print(labelEncoder)
                    for column, le in labelEncoder.items():
                        if column in df.columns:
                            df[column] = le.transform(df[column])

                    # df = pd.get_dummies(df, columns=categorical_columns)
                    df = df.reindex(
                        columns=scaler.feature_names_in_, fill_value=0)
                    data_scaled = scaler.transform(df)
                    return data_scaled

                def make_predictions(model, data_scaled):
                    predictions = model.predict(data_scaled)
                    return predictions

                df = pd.read_csv(datasetUrl)
                column_names = df.columns.drop(df.columns[-1])
                categorical_columns = df.select_dtypes(
                    include=['object']).columns

                data_scaled = preprocess_input(
                    input_data, scaler, labelEncoder, categorical_columns, column_names)
                predictions = make_predictions(model, data_scaled)

                print(predictions)
                return Response({"prediction": predictions.tolist()}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Failed to load model or scaler"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif task == "classification":
            helpers = current_model["helpers"]
            scalerUrl = helpers[0]["scaler"]
            labelEncoderUrl = helpers[1]["label_encoders"]

            model_name = modelUrl.split('/')[-1]
            scaler_name = scalerUrl.split('/')[-1]
            label_encoder_name = labelEncoderUrl.split('/')[-1]

            model_path = os.path.join("models", model_name)
            scaler_path = os.path.join("models", scaler_name)
            label_encoder_path = os.path.join("models", label_encoder_name)

            model = load_model_from_local(model_path)
            scaler = load_scaler_from_local(scaler_path)
            label_encoders = load_label_encoders_from_local(label_encoder_path)

            def preprocess_input(data, scaler, label_encoders, column_names):
                df = pd.DataFrame([data], columns=column_names)
                print(column_names)
                for column, le in label_encoders.items():
                    if column in df.columns:
                        df[column] = le.transform(df[column])

                df = pd.get_dummies(df, columns=df.select_dtypes(
                    include=['object']).columns, drop_first=True)
                df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
                data_scaled = scaler.transform(df)
                return data_scaled

            def make_predictions(model, data_scaled):
                predictions_proba = model.predict(data_scaled)
                if predictions_proba.shape[1] == 1:
                    predictions = (predictions_proba > 0.5).astype(int)
                else:
                    predictions = np.argmax(predictions_proba, axis=1)
                return predictions, predictions_proba

            if model and scaler:
                df = pd.read_csv(datasetUrl)
                column_names = df.columns.drop(df.columns[-1]).tolist()
                categorical_columns = df.drop(
                    columns=df.columns[-1]).select_dtypes(include=['object']).columns.tolist()
                y = df.iloc[:, -1]

                data_scaled = preprocess_input(
                    input_data, scaler, label_encoders, column_names)

                predictions, predictions_proba = make_predictions(
                    model, data_scaled)

                if predictions_proba.shape[1] == 1:
                    pred = None
                    if y.dtype == object:
                        pred = numToText(y, predictions[0][0])
                    else:
                        pred = int(predictions[0][0])
                    return Response({"prediction": [{"predicted_class": pred}, {"probability": float(predictions_proba[0][0])}]}, status=status.HTTP_200_OK)
                else:
                    if y.dtype == object:
                        pred = numToText(y, predictions[0])
                    else:
                        pred = int(predictions[0])
                    return Response({"prediction": [{"predicted_class": pred}, {"probabilities": predictions_proba[0].tolist()}]}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Failed to load model, scaler, or label encoders"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif task == "text":
            helpers = current_model["helpers"]
            tokenizer = helpers[0]["tokenizer"]
            label_encoder = helpers[1]["label_encoder"]

            model_name = modelUrl.split('/')[-1]
            tokenizer_name = tokenizer.split('/')[-1]
            label_encoder_name = label_encoder.split('/')[-1]

            model_path = os.path.join("models", model_name)
            tokenizer_path = os.path.join("models", tokenizer_name)
            label_encoder_path = os.path.join("models", label_encoder_name)

            model = load_model_from_local(model_path)

            def preprocess_text(text, tokenizer, max_sequence_length):
                sequences = tokenizer.texts_to_sequences([text])
                padded_sequences = pad_sequences(
                    sequences, maxlen=max_sequence_length)
                return padded_sequences

            def make_predictions(text, model, tokenizer, label_encoder, max_sequence_length):
                preprocessed_text = preprocess_text(
                    text, tokenizer, max_sequence_length)
                predictions = model.predict(preprocessed_text)
                predicted_class = tf.argmax(predictions, axis=1)
                predicted_label = label_encoder.inverse_transform(
                    predicted_class)
                return predicted_label[0], predictions[0]

            model = load_model_from_local(model_path)
            tokenizer = load_tokenizer(tokenizer_path)
            label_encoder = load_label_encoder(label_encoder_path)

            if model and tokenizer and label_encoder:
                max_sequence_length = 100
                predicted_label, prediction_proba = make_predictions(
                    input_data, model, tokenizer, label_encoder, max_sequence_length)
                print(f"Predicted label: {predicted_label}")
                print(f"Prediction probabilities: {prediction_proba}")

                return Response({"prediction": [{"predicted_class": predicted_label}, {"probabilities": prediction_proba}]}, status=status.HTTP_200_OK)

        elif task == "image":
            model_name = modelUrl.split('/')[-1]
            model_path = os.path.join("models", model_name)
            classNames = current_model["classnames"]
            model = load_model_from_local(model_path)

            if not image_file and not image_url:
                return Response({"error": "Please provide either an image file or an image URL"}, status=status.HTTP_400_BAD_REQUEST)

            model = load_model_from_local(model_path)

            def prepare_image(img, img_height=120, img_width=120):
                img = img.resize((img_width, img_height))
                img_array = image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                return img_array

            def make_predictions(img_array, model):
                predictions = model.predict(img_array)
                # class_idx = np.argmax(predictions, axis=1)[0]
                # class_prob = predictions[0][class_idx]
                score = tf.nn.softmax(predictions[0])
                return tf.argmax(score), tf.reduce_max(score)

            try:
                if image_file:
                    img = Image.open(image_file)
                elif image_url:
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))
                else:
                    return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

                img_array = prepare_image(img)
                class_idx, class_prob = make_predictions(img_array, model)

                return Response({
                    "predictedClassIndex": classNames[int(class_idx)],
                    "classProbability": float(class_prob)
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        elif task == "chatbot":
            if datasetUrl.find(".pdf") != -1:
                pdf_dir = "pdfs"
                pdf_name = datasetUrl.split('/')[-1]
                pdf_path = os.path.join(pdf_dir, pdf_name)
                pdf_document = fitz.open(pdf_path)

                text = ""
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text()

                pdf_document.close()
                response = geminiModel.generate_content(
                    f"Context: {text} Answer the following question in less than 100 words no matter what and if the answer doesnt exist in the context, simple reply with answer not available: {input_data}")

                return Response({
                    "answer": response.text,
                }, status=status.HTTP_200_OK)
            else:
                model_path = os.path.join(
                    "models", "sentence_transformer_model")
                helpers = current_model["helpers"]
                question_embeddings = helpers[0]["question_embeddings"]
                answer_embeddings = helpers[1]["answer_embeddings"]

                question_embeddings_name = question_embeddings.split('/')[-1]
                answer_embeddings_name = answer_embeddings.split('/')[-1]

                question_embeddings_path = os.path.join(
                    "models", question_embeddings_name)
                answer_embeddings_path = os.path.join(
                    "models", answer_embeddings_name)

                model_path = os.path.join(
                    "models", "sentence_transformer_model")
                try:
                    model = SentenceTransformer(model_path)
                    question_embeddings = torch.load(question_embeddings_path)
                    answer_embeddings = torch.load(answer_embeddings_path)
                except Exception as e:
                    return Response({"error": f"Failed to load model or embeddings: {str(e)}"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                data_url = datasetUrl
                try:
                    response = requests.get(data_url)
                    qa_data = json.loads(response.text)
                    questions = [item['question'] for item in qa_data]
                    answers = [item['answer'] for item in qa_data]
                except Exception as e:
                    return Response({"error": f"Failed to load QA data: {str(e)}"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                if not input_data:
                    return Response({"error": "Please provide a question"},
                                    status=status.HTTP_400_BAD_REQUEST)

                try:
                    answer, similarity_percentage = get_answer(
                        input_data, model, question_embeddings, answer_embeddings, answers)
                    return Response({
                        "answer": answer,
                        "similarityPercentage": round(similarity_percentage, 2)
                    }, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({"error": f"Error processing question: {str(e)}"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"error": "Unknown error occurred"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserUpdateView(APIView):
    def put(self, request):
        userId = request.data.get('userId')
        if not userId:
            return Response({"error": "userId is required"}, status=status.HTTP_400_BAD_REQUEST)

        user = get_object_or_404(userSignup, id=userId)
        serializer = UserUpdateSerializer(
            user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return_serializer = signupSerializer(user)
            return Response(return_serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        userId = request.query_params.get('userId')
        if not userId:
            return Response({"error": "userId is required"}, status=status.HTTP_400_BAD_REQUEST)

        user = get_object_or_404(userSignup, id=userId)
        current_date = timezone.now()

        if user.purchase_date + timedelta(days=30) <= current_date:
            user.has_expired = True
            user.save()

        serializer = signupSerializer(user)
        return Response(serializer.data)


class SocketClient:
    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('localhost', 12345))

    def send_epoch_info(self, epoch_info):
        data = json.dumps(epoch_info)
        self.client_socket.send(data.encode())

    def close(self):
        self.client_socket.close()

    def start_server():
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', 12345))
        server_socket.listen(1)

        print("Server is waiting for a connection...")


def isText(df, columns):
    text = []
    for column in columns:
        if df[column].dtype == object:
            text.append(True)
        else:
            text.append(False)

    if all(text):
        return True
    else:
        return False


def textToNum(finalColumn, x):
    arr = finalColumn.unique()
    indices = np.where(arr == x)[0]
    if indices.size > 0:
        index = indices[0]
        return index
    else:
        return -1


class UploadFileView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            file_name = uploaded_file.name
            name = uploaded_file.name.split(
                '.')[0] + str(uuid.uuid4()) + '.' + uploaded_file.name.split('.')[1]
            base = f"https://xanderco-storage.s3-accelerate.amazonaws.com/{file_name}"
            file_path = default_storage.save(name, uploaded_file)
            s3_client = boto3.client('s3', aws_access_key_id="AKIA3F3GRYWMGLSXHLVJ",
                                     aws_secret_access_key="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",
                                     region_name='ap-south-1', config=Config(s3={'use_accelerate_endpoint': True}))

            pre_signed_url = s3_client.generate_presigned_url(
                'put_object',
                Params={'Bucket': 'xanderco-storage', 'Key': file_name},
                ExpiresIn=7200,
                HttpMethod='PUT'
            )
            with open(file_path, 'rb') as f:
                response = requests.put(pre_signed_url, data=f)
            os.remove(file_path)
            if response.status_code == 200:
                return Response({'file_url': base}, status=201)
            else:
                return Response({'error': 'File upload failed', 'details': response.text}, status=response.status_code)
        #     url = f'https://yzeywwy3b6.execute-api.ap-south-1.amazonaws.com/deb/xanderco-storage/{file_name}'

        #     response = requests.put(url, data=uploaded_file.read())

        #     if response.status_code == 200:
        #     else:
        #         return Response({'error': 'File upload failed', 'details': response.text}, status=response.status_code)
        else:
            return Response(serializer.errors, status=400)

class ChatTrainView(APIView):
    serializer_class = ChatTrainSerializer
    api_url = 'https://apiv3.xanderco.in/core/store/'
    
    def determine_task(self, file_path):
        file_type = mimetypes.guess_type(file_path)[0]
        task_type = ''
        architecture_details = ''
        architecture = []
        hyperparameters = {}

        if file_type == 'application/zip' or file_type == "application/x-zip-compressed":
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if any(file.endswith('.mp3') for file in file_list):
                    task_type = 'audio'
                    architecture_details = 'Audio processing architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")
                elif any(file.endswith(('.jpg', '.jpeg', '.png')) for file in file_list):
                    task_type = 'image'
                    architecture_details = 'Image processing architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")
                else:
                    raise ValueError("No supported file types found in ZIP")

        elif file_type == 'application/json' or file_type == 'application/pdf':
            task_type = 'chatbot'
            architecture_details = 'Chatbot architecture'
            architecture, hyperparameters = returnArch(
                arch_data, task_type, "DL", "default")

        else:
            df = pd.read_csv(file_path)
            num_columns = df.select_dtypes(include=[np.number]).shape[1]
            all_columns = list(df.columns)
            final_column = df.iloc[:, -1]

            if isText(df, all_columns) == True and df.apply(lambda col: col.str.len().mean() > 10).any():
                print("Going in")
                task_type = 'text'
                architecture_details = 'NLP architecture'
                architecture, hyperparameters = returnArch(
                    arch_data, task_type, "DL", "default")
            else:
                df[all_columns[-1]] = df[all_columns[-1]
                                         ].apply(lambda x: textToNum(final_column, x))
                final_column = df.iloc[:, -1]
                unique_values = final_column.unique()
                if len(unique_values) / len(final_column) > 0.1:
                    task_type = 'regression'
                    architecture_details = 'Regression model architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")
                else:
                    task_type = 'classification'
                    architecture_details = 'Classification model architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")

        return task_type, hyperparameters, architecture

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data['query']
            userId = serializer.validated_data['userId']
            context = '''
            User will provide you with a query describing the AI model they want to build. You will extract following things from the query:
            1. Type of task, i.e., whether it is Regression, Classification, Image Classification, Textual Classification or Chatbot development.
            2. What kind of model do they want to build like a credit card fraud detection model, a medical chatbot, an image classifier to detect cats or dogs etc. ? Clearly mention what exactly is model is supposed to do.
            3. Have they mentioned what type of architecture they want us to use, for example have they mentioned they want us to use Deep Learning or LSTM layers or maybe SVM or XGBoost etc.? If not mentioned return None
            Just mention the answers separated by comma and no need for numbering.
            '''
            print(query)
            try:
                response = geminiProModel.generate_content(f"Context: {context} Query: {query}")
                arr = response.text.split(',')

                requirements = {
                    "task": arr[0], "model": arr[1].strip(), "architecture": arr[2].strip()
                }

                datasets = kaggle.api.dataset_list(search=requirements["model"])
                dataset_refs = [dataset.ref for dataset in datasets]

                print(datasets)

                dataset_context = f"Model to train: {requirements['model']} Dataset names: {dataset_refs}."
                dataset_query = f"Now based on the model to train name, return from the dataset names array that dataset name that resonates the most with the name of the model to train. You will always return exactly one single name from the dataset names array. And make sure the name is exactly the same as in the dataset names array, i.e., in the format of username/dataset-name!"
                
                dataset_response = geminiProModel.generate_content(
                    f"Context: {dataset_context} Query: {dataset_query}")
                selected_ref = dataset_response.text.strip()

                dataset = next((d for d in datasets if d.ref == selected_ref), None)
                print(dataset)
                if dataset is None:
                    return Response({
                        "status": "error",
                        "message": f"Dataset '{selected_ref}' not found in the search results."
                    }, status=status.HTTP_404_NOT_FOUND)

                kaggle.api.dataset_download_files(dataset.ref, path='./data', unzip=True)
                dirs = os.listdir('data')

                file_path = None
                file_name = ''
                train_result = checkTrain(dirs)
                test_result = checkTest(dirs)

                if train_result["index"] != -1:
                    if not train_result["isDir"]:
                        file_name = remove_whitespace(dirs[train_result["index"]])
                        file_path = os.path.join('data', file_name)
                    else:
                        train_dir = os.path.join('data', dirs[train_result["index"]])
                        file_name = remove_whitespace(os.listdir(train_dir)[0])
                        file_path = os.path.join(train_dir, file_name)
                elif test_result["index"] != -1:
                    if not test_result["isDir"]:
                        file_name = remove_whitespace(dirs[test_result["index"]])
                        file_path = os.path.join('data', file_name)
                    else:
                        test_dir = os.path.join('data', dirs[test_result["index"]])
                        file_name = remove_whitespace(os.listdir(test_dir)[0])
                        file_path = os.path.join(test_dir, file_name)
                else:
                    file_name = remove_whitespace(dirs[0])
                    file_path = os.path.join('data', file_name)

                print(file_name)
                print(file_path)

                name = file_name.split(
                '.')[0] + str(uuid.uuid4()) + '.' + file_name.split('.')[1]
                
                new_file_path = file_path.replace(file_name, name)

                os.rename(file_path, new_file_path)

                file_size_bytes = os.path.getsize(new_file_path)
                file_size_gb = file_size_bytes / (1024 ** 3)
                print(file_size_gb)

                user = get_object_or_404(userSignup, id=userId)
                currUsage = float(user.s3_storage_used)
                user.s3_storage_used = currUsage + file_size_gb
                user.save()

                task_type, hyperparameter, architecture_details = self.determine_task(
                new_file_path)

                with open(new_file_path, 'rb') as file_obj:
                    file = {'file': file_obj}
                    response_model = requests.post(self.api_url, files=file)

                response_data_model = response_model.json()
                cloud_url = response_data_model.get('file_url')
                print(cloud_url)
                
                response_data = {
                                'task_type': task_type,
                                'architecture_details': architecture_details,
                                'cloud_url': cloud_url,
                                'hyperparameter': hyperparameter
                            }
                
                print(response_data)
                if cloud_url:
                    shutil.rmtree('data')
                    print("'data' directory removed successfully.")
                    if type(user.dataset_url) == str:
                        datasets = json.loads(user.dataset_url)
                    else:
                        datasets = user.dataset_url
                    datasets.append(response_data)
                    user.dataset_url = datasets
                    user.save()
                    
                    return Response(response_data, status=status.HTTP_200_OK)
                else:
                    return Response({
                        "status": "error",
                        "message": f"Failed to upload model. Error: {response_data_model.get('error')}"
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            except Exception as e:
                return Response({
                    "status": "error",
                    "message": f"An error occurred: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "status": "error",
            "message": "Invalid data"
        }, status=status.HTTP_400_BAD_REQUEST)



class DatasetUploadView(APIView):
    serializer_class = DatasetUploadSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            name = uploaded_file.name.split(
                '.')[0] + str(uuid.uuid4()) + '.' + uploaded_file.name.split('.')[1]
            userId = serializer.validated_data['userId']
            name = name.replace(" ", "")
            pdf_dir = 'pdfs'
            if not os.path.isdir(pdf_dir):
                os.makedirs(pdf_dir)

            file_path = default_storage.save(name, uploaded_file)

            if (name.find(".pdf") != -1):
                pdf_path = os.path.join(pdf_dir, name)
                pdf_path_new = default_storage.save(pdf_path, uploaded_file)

            file_size_bytes = uploaded_file.size
            file_size_gb = file_size_bytes / (1024 ** 3)

            user = get_object_or_404(userSignup, id=userId)
            currUsage = float(user.s3_storage_used)
            user.s3_storage_used = currUsage + file_size_gb
            user.save()

            if user.max_cpu_hours_allowed > 0 and user.cpu_hours_used >= user.max_cpu_hours_allowed:
                raise PermissionDenied("CPU hours limit reached")

            if user.max_gpu_hours_allowed > 0 and user.gpu_hours_used >= user.max_gpu_hours_allowed:
                raise PermissionDenied("GPU hours limit reached")

            if user.s3_storage_used > 0 and user.s3_storage_used >= user.max_storage_allowed:
                raise PermissionDenied("Storage limit reached")

            if user.has_expired == True:
                raise PermissionDenied("Your plan has expired")

            task_type, hyperparameter, architecture_details = self.determine_task(
                file_path)

            dataset = Dataset.objects.create(
                name=uploaded_file.name,
                size_gb=file_size_gb,
                task_type=task_type,
                architecture_details=architecture_details,
                hyperparameter=hyperparameter
            )

            api_url = 'https://apiv3.xanderco.in/core/store/'

            cloud_url = self.upload_to_s3(api_url, file_path)

            response_data = {
                'task_type': task_type,
                'architecture_details': architecture_details,
                'cloud_url': cloud_url,
                'hyperparameter': hyperparameter
            }
            if type(user.dataset_url) == str:
                datasets = json.loads(user.dataset_url)
            else:
                datasets = user.dataset_url
            datasets.append(response_data)
            user.dataset_url = datasets

            user.save()

            # if name.find(".pdf") == -1:
            os.remove(file_path)

            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def determine_task(self, file_path):
        file_type = mimetypes.guess_type(file_path)[0]
        task_type = ''
        architecture_details = ''
        architecture = []
        hyperparameters = {}

        if file_type == 'application/zip' or file_type == "application/x-zip-compressed":
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if any(file.endswith('.mp3') for file in file_list):
                    task_type = 'audio'
                    architecture_details = 'Audio processing architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")
                elif any(file.endswith(('.jpg', '.jpeg', '.png')) for file in file_list):
                    task_type = 'image'
                    architecture_details = 'Image processing architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")
                else:
                    raise ValueError("No supported file types found in ZIP")

        elif file_type == 'application/json' or file_type == 'application/pdf':
            task_type = 'chatbot'
            architecture_details = 'Chatbot architecture'
            architecture, hyperparameters = returnArch(
                arch_data, task_type, "DL", "default")

        else:
            df = pd.read_csv(file_path)
            num_columns = df.select_dtypes(include=[np.number]).shape[1]
            all_columns = list(df.columns)
            final_column = df.iloc[:, -1]

            if isText(df, all_columns) == True and df.apply(lambda col: col.str.len().mean() > 10).any():
                print("Going in")
                task_type = 'text'
                architecture_details = 'NLP architecture'
                architecture, hyperparameters = returnArch(
                    arch_data, task_type, "DL", "default")
            else:
                df[all_columns[-1]] = df[all_columns[-1]
                                         ].apply(lambda x: textToNum(final_column, x))
                final_column = df.iloc[:, -1]
                unique_values = final_column.unique()
                if len(unique_values) / len(final_column) > 0.1:
                    task_type = 'regression'
                    architecture_details = 'Regression model architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")
                else:
                    task_type = 'classification'
                    architecture_details = 'Classification model architecture'
                    architecture, hyperparameters = returnArch(
                        arch_data, task_type, "DL", "default")

        return task_type, hyperparameters, architecture

    def upload_to_s3(self, endpoint, file_path):
        file = {
            'file': open(file_path, 'rb')
        }
        try:
            response = requests.post(endpoint, files=file)
            response_data = response.json()
            print(response_data)
            if response.status_code == 200 or response.status_code == 201:
                pdf_info = response_data.get('file_url')
                initial_url = pdf_info
                print(f"File uploaded successfully. URL: {initial_url}")
                return initial_url
            else:
                print(
                    f"Failed to upload file. Error: {response_data.get('error')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None


class TrainModelView(APIView):
    def post(self, request):
        serializer = TaskSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            dataset_url = data['dataset_url']
            hasChanged = data['hasChanged']
            task = data['task']
            mainType = data['mainType']
            archType = data['archType']
            userId = data['userId']
            architecture = {}
            hyperparameters = data["hyperparameters"]

            user = get_object_or_404(userSignup, id=userId)

            if user.max_cpu_hours_allowed > 0 and user.cpu_hours_used >= user.max_cpu_hours_allowed:
                raise PermissionDenied("CPU hours limit reached")

            if user.max_gpu_hours_allowed > 0 and user.gpu_hours_used >= user.max_gpu_hours_allowed:
                raise PermissionDenied("GPU hours limit reached")

            if user.s3_storage_used > 0 and user.s3_storage_used >= user.max_storage_allowed:
                raise PermissionDenied("Storage limit reached")

            plan = user.plan

            if type(user.trained_model_url) == str:
                datasets = json.loads(user.trained_model_url)
            else:
                datasets = user.trained_model_url

            start_time = time.time()
            channel_layer = get_channel_layer()

            if task == "regression" and not hasChanged:
                if mainType == "DL":
                    architecture = returnArch(
                        arch_data, task, mainType, archType)
                    model_trainer = RegressionDL(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId)
                    executor = model_trainer.execute()

                    for epoch_info in executor:
                        if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )
                            print(
                                f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, Test Loss: {epoch_info['test_loss']:.4f}")
                        else:
                            model_obj = epoch_info
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )
                            print("Final model object:", epoch_info)
                            break

                elif mainType == "ML":
                    print(dataset_url, hasChanged, task, mainType, archType,)
                    architecture = returnArch(
                        arch_data, task, mainType, archType)

                    model_trainer = RegressionML(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    print(dataset_url, hasChanged, task, mainType,
                          archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()

            elif task == "regression" and hasChanged:
                architecture = data['arch_data'].get('architecture', [])
                hyperparameters = data['arch_data'].get('hyperparameters', {})
                if mainType == "DL":
                    model_trainer = RegressionDL(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId)
                    model_obj = model_trainer.execute()
                    for epoch_info in executor:
                        if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )
                            print(
                                f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, Test Loss: {epoch_info['test_loss']:.4f}")
                        else:
                            model_obj = epoch_info
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )
                            print("Final model object:", epoch_info)
                            break

                elif mainType == "ML":
                    model_trainer = RegressionML(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()

            if task == "classification" and not hasChanged:
                if mainType == "DL":
                    architecture = returnArch(
                        arch_data, task, mainType, archType)
                    model_trainer = ClassificationDL(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId)
                    executor = model_trainer.execute()
                    for epoch_info in executor:
                        if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )

                        else:
                            model_obj = epoch_info
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )
                            print("Final model object:", epoch_info)
                            break
                elif mainType == "ML":
                    architecture, hyperparameters = returnArch(
                        arch_data, task, mainType, archType)
                    model_trainer = ClassificationML(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()

            elif task == "classification" and hasChanged:
                if mainType == "DL":
                    model_trainer = ClassificationDL(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId)
                    executor = model_trainer.execute()

                    for epoch_info in executor:
                        if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )

                        else:
                            model_obj = epoch_info
                            async_to_sync(channel_layer.group_send)(
                                f"user_{userId}",
                                {
                                    'type': 'send_update',
                                    'text': json.dumps(epoch_info)
                                }
                            )
                            print("Final model object:", epoch_info)
                            break

                elif mainType == "ML":
                    architecture, hyperparameters = returnArch(
                        arch_data, task, mainType, archType)
                    model_trainer = ClassificationML(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()

            if task == "image":
                architecture = returnArch(
                    arch_data, task, mainType, archType)

                trainer = ImageModelTrainer(
                    dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId)
                executor = trainer.execute()
                for epoch_info in executor:
                    if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                        async_to_sync(channel_layer.group_send)(
                            f"user_{userId}",
                            {
                                'type': 'send_update',
                                'text': json.dumps(epoch_info)
                            }
                        )

                    else:
                        model_obj = epoch_info
                        async_to_sync(channel_layer.group_send)(
                            f"user_{userId}",
                            {
                                'type': 'send_update',
                                'text': json.dumps(epoch_info)
                            }
                        )
                        print("Final model object:", epoch_info)
                        break

                if model_obj:
                    print(f"Model Object: {model_obj}")
                else:
                    print("Failed to train and upload the model.")

            if task == "text":
                architecture = returnArch(
                    arch_data, task, mainType, archType)
                print(architecture)
                model = TextModel(
                    dataset_url=dataset_url,
                    hasChanged=hasChanged,
                    task='text',
                    mainType=mainType,
                    archType=archType,
                    architecture=architecture,
                    hyperparameters=hyperparameters,
                    userId=userId
                )
                executor = model.execute()

                for epoch_info in executor:
                    # send this epoch info via sockets
                    if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                        # socket_client.send_epoch_info(epoch_info)
                        print(epoch_info)
                        async_to_sync(channel_layer.group_send)(
                            f"user_{userId}",
                            {
                                'type': 'send_update',
                                'text': json.dumps(epoch_info)
                            }
                        )

                    else:
                        model_obj = epoch_info
                        async_to_sync(channel_layer.group_send)(
                            f"user_{userId}",
                            {
                                'type': 'send_update',
                                'text': json.dumps(epoch_info)
                            }
                        )
                        print("Final model object:", epoch_info)
                        break

            if task == "chatbot":
                if dataset_url.find("pdf") != -1:
                    model = ChatbotPDF(
                        dataset_url=dataset_url,
                        task='chatbot',
                        userId=userId,
                    )
                    model_obj = model.execute()
                else:
                    model = Chatbot(
                        dataset_url=dataset_url,
                        hasChanged=hasChanged,
                        task='chatbot',
                        mainType=mainType,
                        archType=archType,
                        architecture=architecture,
                        hyperparameters=hyperparameters,
                        userId=userId
                    )

                    model_obj = model.execute()

            end_time = time.time()
            deltaTime = (end_time - start_time) / (60 * 60)

            datasets.append(model_obj)
            size = model_obj["size"]
            cpu_hours_used = float(user.cpu_hours_used)
            gpu_hours_used = float(user.gpu_hours_used)

            if plan == "free" or plan == "researcher" or plan == "basic":
                cpu_hours_used += deltaTime
                user.cpu_hours_used = cpu_hours_used
            else:
                gpu_hours_used += deltaTime
                print(gpu_hours_used)
                print(user.gpu_hours_used)
                user.gpu_hours_used = gpu_hours_used
                user.save()

            user.s3_storage_used = float(user.s3_storage_used) + size
            user.trained_model_url = datasets

            user.save()
            # socket_client.close()

            result_serializer = ResultSerializer(model_obj)
            return Response(result_serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
