# storage/views.py
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
from .serializers import DatasetUploadSerializer, DatasetSerializer,signupSerializer
from .models import Dataset, S3StorageUsage,userSignup
import requests

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

            task_type, architecture_details = self.determine_task(file_path)
            print(file_path)

            dataset = Dataset.objects.create(
                name=uploaded_file.name,
                size_gb=file_size_gb,
                task_type=task_type,
                architecture_details=architecture_details
            )
            
            api_url = 'https://s3-api-uat.idesign.market/api/upload'
            bucket_name = 'idesign-quotation'

            cloud_url = self.upload_to_s3(api_url, bucket_name, file_path)

            response_data = {
                'task_type': task_type,
                'architecture_details': architecture_details,
                'cloud_url': cloud_url,
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def determine_task(self, file_path):
        file_type = mimetypes.guess_type(file_path)[0]
        task_type = ''
        architecture_details = ''

        if file_type == 'application/zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                if any(file.endswith('.mp3') for file in file_list):
                    task_type = 'Audio'
                    architecture_details = 'Audio processing architecture'
                elif any(file.endswith(('.jpg', '.jpeg', '.png')) for file in file_list):
                    task_type = 'Image'
                    architecture_details = 'Image processing architecture'
        elif file_type == 'application/json':
            task_type = 'JSON'
            architecture_details = 'Chatbot architecture'
        else:
            df = pd.read_csv(file_path)
            num_columns = df.select_dtypes(include=[np.number]).shape[1]
            final_column = df.iloc[:, -1]
            if final_column.dtype in [np.float64, np.int64]:
                unique_values = final_column.unique()
                if len(unique_values) / len(final_column) > 0.1:
                    task_type = 'Regression'
                    architecture_details = 'Regression model architecture'
                else:
                    task_type = 'Classification'
                    architecture_details = 'Classification model architecture'
            elif final_column.dtype == object and df.apply(lambda col: col.str.len().mean() > 10).any():
                task_type = 'Textual'
                architecture_details = 'NLP architecture'
        os.remove(file_path)
        return task_type, architecture_details
    
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