import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
import joblib
import requests
import os
import uuid
import queue
import threading
import random

class ClassificationDL:
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
        self.model_path = f'model{str(uuid.uuid4())}.h5'
        self.scaler_path = f'scaler{str(uuid.uuid4())}.pkl'
        
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        self.df = pd.read_csv(self.dataset_url)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        
        self.X = pd.get_dummies(self.X, drop_first=True)

    def preprocess_data(self):
        self.scaler = StandardScaler()
        self.X_standardized = self.scaler.fit_transform(self.X)
        self.X = self.X_standardized
        self.y = self.y.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
    def create_model(self):
        self.model = tf.keras.Sequential()
        if self.task == "classification" and self.hasChanged:
            for arch in self.architecture:
                if arch['layer'] == "Dense" and arch['define_input_shape'] == "true":
                    self.model.add(Dense(arch['neurons'], input_shape=(self.X_train.shape[1], ), activation=arch['activation']))
                elif arch['layer'] == "Dense" and arch['define_input_shape'] == "false":
                    self.model.add(Dense(arch['neurons'], activation=arch['activation']))
                elif arch['layer'] == "Dropout":
                    self.model.add(Dropout(arch['ratio']))
        else:
            self.model.add(Dense(128, input_shape=(self.X_train.shape[1], ), activation="relu"))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(32 ,activation="relu"))
            self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def train_model(self):
        self.epoch_data = []

        class CustomCallback(Callback):
            def __init__(self, outer_instance, validation_data):
                super().__init__()
                self.outer_instance = outer_instance
                self.validation_data = validation_data

            def on_epoch_end(self, epoch, logs=None):
                test_loss, test_acc = self.model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0)
                train_loss, train_acc = logs['loss'], logs['accuracy']
                
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc
                }
                
                self.outer_instance.epoch_data.append(epoch_info)
                self.outer_instance.current_epoch_info = epoch_info
                self.outer_instance.epoch_info_queue.put(epoch_info)

        self.epoch_info_queue = queue.Queue()
        custom_callback = CustomCallback(self, validation_data=(self.X_test, self.y_test))
        
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.hyperparameters['epochs'],
            batch_size=self.hyperparameters['batch_size'],
            validation_data=(self.X_test, self.y_test),
            callbacks=[custom_callback],
            verbose=0
        )
        
    def evaluate_model(self):
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        # print(f"Accuracy: {self.accuracy}")
        
    def save_model(self):
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
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
        
        training_thread = threading.Thread(target=self.train_model)
        training_thread.start()

        for _ in range(self.hyperparameters['epochs']):
            yield self.epoch_info_queue.get()

        training_thread.join()

        self.evaluate_model()
        self.save_model()
        model_url, scaler_url = self.upload_files_to_api()
        
        _id = str(uuid.uuid4())

        df = pd.read_csv(self.dataset_url)
        data = df.iloc[int(random.random() * len(df.values.tolist()))].tolist()[0:-1]
        formatted_dat = [f"'{item}'" if isinstance(item, str) else str(item) for item in data]
        interference_code = f'''
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import requests
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_url = '{self.dataset_url}'
model_url = '{model_url}'
scaler_url = '{scaler_url}'
input_data = [{', '.join(formatted_dat)}] # your input data

def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download",  url, ":", response.status_code)

def load_model_from_local(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        return None

def load_scaler(url):
    response = requests.get(url)
    if response.status_code == 200:
        scaler_content = response.content
        try:
            scaler = joblib.load(io.BytesIO(scaler_content))
            return scaler
        except Exception as e:
            return None
    else:
        return None

def preprocess_input(data, scaler, categorical_columns, column_names):
    df = pd.DataFrame([data], columns=column_names)
    df = pd.get_dummies(df, columns=categorical_columns)
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    data_scaled = scaler.transform(df)
    return data_scaled

def make_predictions(model, data_scaled):
    predictions_proba = model.predict(data_scaled)
    predictions = (predictions_proba > 0.5).astype(int)
    return predictions, predictions_proba

if __name__ == "__main__":
    model_path = model_url.split('/')[-1]
    download_file(model_url, model_path)
    scaler = load_scaler(scaler_url)

    model = load_model_from_local(model_path)

    if model and scaler:
        df = pd.read_csv(dataset_url)
        column_names = df.columns.drop(df.columns[-1])
        categorical_columns = df.select_dtypes(include=['object']).columns

        data_scaled = preprocess_input(input_data, scaler, categorical_columns, column_names)

        predictions, predictions_proba = make_predictions(model, data_scaled)

        print("Predicted class: ", predictions[0][0])
        print("Prediction probability:", predictions_proba[0][0])
    else:
        print("Failed to load model or scaler.")
        '''

        model_obj = {
            "modelUrl": model_url if model_url and scaler_url else "",
            "size": os.path.getsize(self.model_path) / (1024 ** 3) if model_url and scaler_url else 0,
            "id": _id if model_url and scaler_url else "",
            "helpers": [{"scaler": scaler_url}] if model_url and scaler_url else [],
            "modelArch": self.architecture,
            "hyperparameters": self.hyperparameters,
            "epoch_data": self.epoch_data,
            "task": self.task,
            "interferenceCode": interference_code
        }
        os.remove(self.model_path)
        os.remove(self.scaler_path)
        yield model_obj if model_url and scaler_url else None
