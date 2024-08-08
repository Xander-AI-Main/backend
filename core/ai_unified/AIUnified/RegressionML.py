import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import os
import uuid

class RegressionML:
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
        
        self.load_data()
        self.preprocess_data()

    def is_regression(self, y):
        unique_elements = len(set(y))
        return unique_elements / len(y) > 0.1
    
    def load_data(self):
        self.df = pd.read_csv(self.dataset_url)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        
    def preprocess_data(self):
        self.df = self.df.dropna(thresh=len(self.df) * 0.8, axis=1)
        self.df = self.df.dropna()

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df = self.df[(np.abs(self.df[numeric_cols] - self.df[numeric_cols].mean()) <= (3 * self.df[numeric_cols].std())).all(axis=1)]

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        print(self.X)
        print(self.y)

        self.X = pd.get_dummies(self.X, drop_first=True)

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)

    def create_model(self):
        if self.task == "regression":
            if self.archType == '1':
                print("Using Linear Regression")
                self.model = LinearRegression(**self.hyperparameters)
            elif self.archType == "2":
                print("Using XGBoost Regression")
                self.model = xgb.XGBRegressor(**self.hyperparameters)
            elif self.archType == "3":
                print("Using AdaBoost Regression")
                self.model = AdaBoostRegressor(**self.hyperparameters)
            else:
                raise ValueError(f"Unsupported model type: {self.archType}")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, y_pred)
        print(f"MSE: {self.mse}")
        return self.mse
        
    def save_model(self):
        model_filename = f'model{str(uuid.uuid4())}.pkl'
        scaler_filename = f'scaler{str(uuid.uuid4())}.pkl'
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        
        self.model_path = model_filename
        self.scaler_path = scaler_filename
        
    def upload_files_to_api(self):
        try:
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.model_path, 'rb')
            }
            response_model = requests.put(self.api_url, files=files)
            response_data_model = response_model.json()
            model_url = response_data_model.get('locations', [])[0] if response_model.status_code == 200 else None
            
            if model_url:
                print(f"Model uploaded successfully. URL: {model_url}")
            else: 
                print(f"Failed to upload model. Error: {response_data_model.get('error')}")
                return None, None
            
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.scaler_path, 'rb')
            }
            response_scaler = requests.put(self.api_url, files=files)
            response_data_scaler = response_scaler.json()
            scaler_url = response_data_scaler.get('locations', [])[0] if response_scaler.status_code == 200 else None
            
            if scaler_url:
                print(f"Scaler uploaded successfully. URL: {scaler_url}")
            else:
                print(f"Failed to upload scaler. Error: {response_data_scaler.get('error')}")
                return model_url, None
            
            return model_url, scaler_url
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None, None
        
    def execute(self):
        self.create_model()
        self.train_model()
        mse = self.evaluate_model()
        self.save_model()
        model_url, scaler_url = self.upload_files_to_api()
        
        if model_url and scaler_url:
            _id = str(uuid.uuid4())
            model_obj = {
                "modelUrl": model_url,
                "size": os.path.getsize(self.model_path) / (1024 ** 3),
                "id": _id,
                "helpers": [{"scaler": scaler_url}],
                "modelArch": self.archType,
                "hyperparameters": self.hyperparameters,
                "task": self.task_type,
                "datasetUrl": self.dataset_url,
                "epoch_data": [{"epoch": 1, "test_loss": mse}]
            }
            return model_obj
        else:
            return None
