import os
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
from .serializers import DatasetUploadSerializer, DatasetSerializer, signupSerializer, TaskSerializer, ResultSerializer, InterferenceSerializer
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
from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import userSignup
from .serializers import signupSerializer, LoginSerializer, UserUpdateSerializer
from rest_framework.authtoken.models import Token
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAuthenticated
import time
import socket
import threading
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import socketio
import uuid
import random
import numpy as np
import tensorflow as tf
import joblib
import requests
import io
import pandas as pd
import gridfs

def returnArch(data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]

url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/arch.json'

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

class Interference(APIView):
    def post(self, request):
        user_id = request.data.get('userId')
        model_id = request.data.get('modelId')
        modelPath = request.data.get("modelPath")
        scalerPath = request.data.get("scalerPath")
        tokenizerPath = request.data.get("tokenizerPath")
        labelEncoderPath = request.data.get("labelEncoderPath")
        questionEmbeddingPath = request.data.get("questionEmbeddingPath")
        answerEmbeddingPath = request.data.get("answerEmbeddingPath")

        user = get_object_or_404(userSignup, id=user_id)
        
        return_serializer = signupSerializer(user)
        data = return_serializer.data
        
        current_model = next((model for model in data["trained_model_url"] if model["id"] == model_id), None)
        task = current_model["task"]
        datasetUrl = task["datasetUrl"]

        if not modelPath:
            return Response({"error": "Model path is pequired"}, status=status.HTTP_400_BAD_REQUEST)
        
        if task == "regression":
            df = pd.read_csv(self.dataset_url)
            data = df.iloc[int(random.random() *
                            len(df.values.tolist()))].tolist()[0:-1]
            formatted_dat = [f"'{item}'" if isinstance(
                item, str) else str(item) for item in data]

        

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


sio = socketio.Client()
def isText (df, columns):
    text = True
    for column in columns:
        if df[column].dtype == object:
            text = True
        else: 
            text = False
    return text

def textToNum(finalColumn, x):
    arr = finalColumn.unique()
    indices = np.where(arr == x)[0]
    if indices.size > 0:
        index = indices[0]
        return index
    else:
        return -1  

class DatasetUploadView(APIView):
    serializer_class = DatasetUploadSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            print(uploaded_file.name)
            name = uploaded_file.name.split('.')[0] + str(uuid.uuid4()) + '.' + uploaded_file.name.split('.')[1]
            userId = serializer.validated_data['userId']
            file_path = default_storage.save(name, uploaded_file)

            file_size_bytes = uploaded_file.size
            file_size_gb = file_size_bytes / (1024 ** 3)

            user = get_object_or_404(userSignup, id=userId)
            currUsage = float(user.s3_storage_used)
            user.s3_storage_used = currUsage + file_size_gb
            user.save()

            # s3_storage, created = S3StorageUsage.objects.get_or_create(id=1)
            # s3_storage.used_gb += file_size_gb
            # s3_storage.save()
            print(file_path)
            task_type, hyperparameter, architecture_details = self.determine_task(
                file_path)

            dataset = Dataset.objects.create(
                name=uploaded_file.name,
                size_gb=file_size_gb,
                task_type=task_type,
                architecture_details=architecture_details,
                hyperparameter=hyperparameter
            )

            api_url = 'https://s3-api-uat.idesign.market/api/upload'
            bucket_name = 'idesign-quotation'

            cloud_url = self.upload_to_s3(api_url, bucket_name, file_path)

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
            print(datasets)
            datasets.append(response_data)
            user.dataset_url = datasets
            user.save()

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

        elif file_type == 'application/json':
            task_type = 'chatbot'
            architecture_details = 'Chatbot architecture'
            architecture, hyperparameters = returnArch(
                arch_data, task_type, "DL", "default")

        else:
            df = pd.read_csv(file_path)
            num_columns = df.select_dtypes(include=[np.number]).shape[1]
            all_columns = list(df.columns)
            print(all_columns[-1])
            final_column = df.iloc[:, -1]

            print(final_column.dtype)

            if isText(df, all_columns) == True and df.apply(lambda col: col.str.len().mean() > 10).any():
                task_type = 'text'
                architecture_details = 'NLP architecture'
                architecture, hyperparameters = returnArch(
                    arch_data, task_type, "topic classification", "default")
            else:
                df[all_columns[-1]] = df[all_columns[-1]].apply(lambda x: textToNum(final_column, x))
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

    def upload_to_s3(self, endpoint, bucket_name, file_path):
        files = {
            'bucketName': (None, bucket_name),
            'files': open(file_path, 'rb')
        }
        try:
            print(1)
            response = requests.put(endpoint, files=files)
            response_data = response.json()
            print(2)
            if response.status_code == 200:
                pdf_info = response_data.get('locations', [])[0]
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
            architecture = {}  # extract from arch.json
            hyperparameters = data['hyperparameters']

            user = get_object_or_404(userSignup, id=userId)
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
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
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
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
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
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
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
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
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
                    dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
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
                    hyperparameters=hyperparameters
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
                model = Chatbot(
                    dataset_url=dataset_url,
                    hasChanged=hasChanged,
                    task='chatbot',
                    mainType=mainType,
                    archType=archType,
                    architecture=architecture,
                    hyperparameters=hyperparameters
                )

                model_obj = model.execute()

            end_time = time.time()
            deltaTime = (end_time - start_time) / (60 * 60)

            print(model_obj)
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
    
