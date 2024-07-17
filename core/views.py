import os
import zipfile
import pandas as pd
import json
import mimetypes
import boto3
import numpy as np
from django.conf import settings
from rest_framework import viewsets
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import DatasetUploadSerializer, DatasetSerializer,signupSerializer,TaskSerializer, ResultSerializer
from .models import Dataset, S3StorageUsage,userSignup
import requests
import json
import pandas as pd
from .ai_unified.AIUnified.RegressionML import RegressionML
from .ai_unified.AIUnified.RegressionDL import RegressionDL
from .ai_unified.AIUnified.ClassificationDL import ClassificationDL
from .ai_unified.AIUnified.ClassificationML import MLTrainer
from .ai_unified.AIUnified.ImageModelTrainer import ImageModelTrainer
from .ai_unified.AIUnified.TextModel import TextModel
from .ai_unified.AIUnified.Chatbot import Chatbot 


def returnArch (data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if  i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]
        


url = 'https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/arch.json'

try:
    # Fetching the JSON data from the URL
    response = requests.get(url)
    
    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Loading JSON data from the response content
        arch_data= json.loads(response.content)
        
       
    else:
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from {url}: {e}")
    


class DatasetUploadView(APIView):
    serializer_class = DatasetUploadSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            file_path = default_storage.save(uploaded_file.name, uploaded_file)

            file_size_bytes = uploaded_file.size
            file_size_gb = file_size_bytes / (1024 ** 3)

            s3_storage, created = S3StorageUsage.objects.get_or_create(id=1)
            s3_storage.used_gb += file_size_gb
            s3_storage.save()

            task_type,hyperparameter, architecture_details = self.determine_task(file_path)
            print(file_path)

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
                'hyperparameter':hyperparameter
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def determine_task(self, file_path):
        file_type = mimetypes.guess_type(file_path)[0]
        task_type = ''
        architecture_details = ''
        architecture=[]
        hyperparameter={}

        if file_type == 'application/zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if any(file.endswith('.mp3') for file in file_list):
                    task_type = 'Audio'
                    architecture_details = 'Audio processing architecture'
                    architecture, hyperparameters = returnArch(arch_data, task_type, "DL", "default")

                elif any(file.endswith(('.jpg', '.jpeg', '.png')) for file in file_list):
                    task_type = 'Image'
                    architecture_details = 'Image processing architecture'
                    architecture, hyperparameters = returnArch(arch_data, task_type, "DL", "default")

        elif file_type == 'application/json':
            task_type = 'JSON'
            architecture_details = 'Chatbot architecture'
            architecture, hyperparameters = returnArch(arch_data, task_type, "DL", "default")

        else:
            df = pd.read_csv(file_path)
            num_columns = df.select_dtypes(include=[np.number]).shape[1]
            final_column = df.iloc[:, -1]
            if final_column.dtype in [np.float64, np.int64]:
                unique_values = final_column.unique()
                if len(unique_values) / len(final_column) > 0.1:
                    task_type = 'regression'
                    architecture_details = 'Regression model architecture'
                    architecture, hyperparameters = returnArch(arch_data, task_type, "DL", "default")
                     
                else:
                    task_type = 'classification'
                    architecture_details = 'Classification model architecture'
                    architecture, hyperparameters = returnArch(arch_data, task_type, "DL", "default")                    
            elif final_column.dtype == object and df.apply(lambda col: col.str.len().mean() > 10).any():
                task_type = 'text'
                architecture_details = 'NLP architecture'
                architecture, hyperparameters = returnArch(arch_data, task_type, "topic classification", "default")

        
        return task_type,hyperparameter,architecture
    
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
                print(f"Failed to upload file. Error: {response_data.get('error')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None

class signupViewset(viewsets.ModelViewSet):
    queryset=userSignup.objects.all()
    serializer_class=signupSerializer

def returnArch(data, task, mainType, archType):
    current_task = arch_data[task]
    for i in current_task:
        if i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]

class TrainModelView(APIView):
    def post(self, request):
        serializer = TaskSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            print(data)
            dataset_url = data['dataset_url']
            hasChanged = data['hasChanged']
            task = data['task']
            mainType = data['mainType']
            archType = data['archType']
            # arch_data = data.get('arch_data', {})
            architecture = {} # extract from arch.json
            hyperparameters = {}
            uploaded_urls = {}
            files_to_upload = ['sentence_transformer_model.zip', 'question_embeddings.pt', 'answer_embeddings.pt']


            if task == "regression" and not hasChanged:

                if mainType == "DL":
                    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
                    model_trainer = RegressionDL(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
                elif mainType == "ML":
                    print(dataset_url, hasChanged, task, mainType, archType,)
                    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
                   
                    model_trainer = RegressionML(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    print(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
            elif task == "regression" and hasChanged:
                architecture = data['arch_data'].get('architecture', [])
                hyperparameters = data['arch_data'].get('hyperparameters', {})
                if mainType == "DL":
                    model_trainer = RegressionDL(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
                elif mainType == "ML":
                    model_trainer = RegressionML(
                        dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
            
            if task == "classification" and not hasChanged:
                if mainType == "DL":
                    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
                    model_trainer = ClassificationDL(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
                elif mainType == "ML":
                    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
                    model_trainer = MLTrainer(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
            elif task == "classification" and hasChanged:
                if mainType == "DL":
                    model_trainer = ClassificationDL(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
                elif mainType == "ML":
                    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
                    model_trainer = MLTrainer(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                    model_obj = model_trainer.execute()
                
            if task=="image":
                architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)

                trainer = ImageModelTrainer(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
                model_obj = trainer.execute()

                if model_obj:
                    print(f"Model Object: {model_obj}")
                else:
                    print("Failed to train and upload the model.")
                        
            if task=="text":
                model = TextModel(
                    dataset_url='train.csv',
                    hasChanged=False,
                    task='text',
                    mainType='topic classification',
                    archType='Default',
                    architecture=architecture,
                    hyperparameters=hyperparameters
                )

                model_obj = model.execute()
                print(model_obj)
            
            if task=="chatbot":
                
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

            result_serializer = ResultSerializer({'model_obj': model_obj})
            return Response(result_serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        
        