import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import requests
import os
import uuid
import queue
import threading
import random

class RegressionDL:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId):
        self.dataset_url = dataset_url
        self.archType = archType
        self.mainType = mainType
        self.architecture = architecture
        self.hasChanged = hasChanged
        self.hyperparameters = hyperparameters
        self.data, self.target_col = self.load_and_prepare_data()
        self.task_type = task
        self.model = None
        self.scaler = StandardScaler()
        self.model_file_path = f'model{str(uuid.uuid4())}.h5'
        self.scaler_file_path = f'scaler{str(uuid.uuid4())}.pkl'
        self.directory_path = "models"
        self.complete_model_path = os.path.join(self.directory_path, self.model_file_path)
        self.complete_scaler_path = os.path.join(self.directory_path, self.scaler_file_path)
        self.api_url = 'https://s3-api-uat.idesign.market/api/upload'
        self.bucket_name = 'idesign-quotation'
        self.epoch_info_queue = queue.Queue()
        self.userId = userId

    def apply_preprocessing(self, value, data):
        # Preprocessing
        uni = list(set(data))
        return uni.index(value)

    def load_and_prepare_data(self):
        df = pd.read_csv(self.dataset_url)
        target_col = df.columns[-1]
        for column_name in df.columns:
            unique_values = df[column_name].nunique()
            print(unique_values)
            if unique_values <= 4 and df[column_name].dtype in ['object', 'string']:
                dummies = pd.get_dummies(df[column_name], prefix=column_name)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[column_name])
                print(f"Dummies created for column '{column_name}'.")

        string_columns = df.select_dtypes(include=['object']).columns
        df = df.drop(columns=string_columns)
        print(df)
        return df, target_col

    def determine_task_type(self):
        target_values = self.data[self.target_col]
        unique_values = set(target_values)
        if len(unique_values) / len(self.data) > 0.1:
            return 'regression'
        else:
            return 'classification'

    def build_model(self):
        input_shape = (self.data.shape[1] - 1,)
        if self.archType == 'default':
            self.model = self.build_dense_model(input_shape)
        # elif self.archType == '4':
        #     self.build_cnn_model(input_shape)
        # elif self.archType == '5':
        #     self.build_lstm_model(input_shape)
        # elif self.archType == '6':
        #     self.build_attention_model(input_shape)
        else:
            raise ValueError(
                "Unsupported model type. Choose from 'dense', 'cnn', 'lstm', 'attention'.")

    def build_dense_model(self, input_shape, layers=[256, 128, 64, 32], activation='relu', output_activation=None):
        model = Sequential()
        model.add(Dense(layers[0], input_shape=input_shape, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.4))
        for layer in layers[1:]:
            model.add(Dense(layer, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(tf.keras.layers.Dropout(0.4))
        model.add(Dense(1, activation=output_activation))
        return model

    # def build_cnn_model(self, input_shape, conv_layers=[(32, (3, 3)), (64, (3, 3))], dense_layers=[64], activation='relu', output_activation=None):
    #     model = Sequential()
    #     model.add(Input(shape=input_shape))
    #     for filters, kernel_size in conv_layers:
    #         model.add(Conv2D(filters, kernel_size, activation=activation))
    #     model.add(Flatten())
    #     for layer in dense_layers:
    #         model.add(Dense(layer, activation=activation))
    #     if self.task_type == 'regression':
    #         model.add(Dense(1, activation=output_activation))
    #     else:
    #         model.add(Dense(len(np.unique(self.data[self.target_col])), activation='softmax'))
    #     self.model = model
    #     return model

    # def build_lstm_model(self, input_shape, lstm_units=50, dense_layers=[64], activation='relu', output_activation=None):
    #     model = Sequential()
    #     model.add(LSTM(lstm_units, input_shape=input_shape, activation=activation))
    #     for layer in dense_layers:
    #         model.add(Dense(layer, activation=activation))
    #     if self.task_type == 'regression':
    #         model.add(Dense(1, activation=output_activation))
    #     else:
    #         model.add(Dense(len(np.unique(self.data[self.target_col])), activation='softmax'))
    #     self.model = model
    #     return model

    # def build_attention_model(self, input_shape, attention_units=50, dense_layers=[64], activation='relu', output_activation=None):
    #     inputs = Input(shape=input_shape)
    #     attention = Attention()([inputs, inputs])
    #     flatten = Flatten()(attention)
    #     x = flatten
    #     for layer in dense_layers:
    #         x = Dense(layer, activation=activation)(x)
    #     if self.task_type == 'regression':
    #         outputs = Dense(1, activation=output_activation)(x)
    #     else:
    #         outputs = Dense(len(np.unique(self.data[self.target_col])), activation='softmax')(x)
    #     model = tf.keras.Model(inputs, outputs)
    #     self.model = model
    #     return model

    def compile_and_train(self):
        if not self.model:
            raise ValueError(
                "Model is not defined. Please build a model first.")

        optimizer = 'adam'
        epochs = int(self.hyperparameters['epochs'])
        batch_size = int(self.hyperparameters['batch_size'])

        if self.task_type == 'regression':
            loss = 'mean_squared_error'
            metrics = []
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])

        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        X = self.scaler.fit_transform(X)

        print(X)
        print(y)

        self.epoch_data = []

        class CustomCallback(Callback):
            def __init__(self, outer_instance, validation_data, task_type):
                super().__init__()
                self.outer_instance = outer_instance
                self.validation_data = validation_data
                self.task_type = task_type

            def on_epoch_end(self, epoch, logs=None):
                test_loss = logs.get('val_loss')
                train_loss = logs.get('loss')
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss
                }

                self.outer_instance.epoch_data.append(epoch_info)
                self.outer_instance.current_epoch_info = epoch_info
                self.outer_instance.epoch_info_queue.put(epoch_info)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.epoch_info_queue = queue.Queue()
        custom_callback = CustomCallback(self, validation_data=(
            X_test, y_test), task_type=self.task_type)

        self.history = self.model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[custom_callback], verbose=0)
        self.X_test, self.y_test = X_test, y_test

    def evaluate_model(self):
        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Evaluation results: {results}")
        return results

    def save_model(self):
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            model_path = os.path.join(self.directory_path, self.model_file_path)
            scaler_path = os.path.join(self.directory_path, self.scaler_file_path)
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
        else:
            model_path = os.path.join(self.directory_path, self.model_file_path)
            scaler_path = os.path.join(self.directory_path, self.scaler_file_path)
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)

    def upload_files_to_api(self):
        try:
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.complete_model_path, 'rb')
            }
            response_model = requests.put(self.api_url, files=files)
            response_data_model = response_model.json()
            model_url = response_data_model.get('locations', [])[
                0] if response_model.status_code == 200 else None

            if model_url:
                print(f"Model uploaded successfully. URL: {model_url}")
            else:
                print(
                    f"Failed to upload model. Error: {response_data_model.get('error')}")
                return None, None

            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.complete_scaler_path, 'rb')
            }
            response_scaler = requests.put(self.api_url, files=files)
            response_data_scaler = response_scaler.json()
            scaler_url = response_data_scaler.get(
                'locations', [])[0] if response_scaler.status_code == 200 else None

            if scaler_url:
                print(f"Scaler uploaded successfully. URL: {scaler_url}")
            else:
                print(
                    f"Failed to upload scaler. Error: {response_data_scaler.get('error')}")
                return model_url, None

            return model_url, scaler_url

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None, None

    def execute(self):
        self.build_model()

        training_thread = threading.Thread(target=self.compile_and_train)
        training_thread.start()

        epochs_completed = 0
        while epochs_completed < int(self.hyperparameters['epochs']):
            try:
                epoch_info = self.epoch_info_queue.get(timeout=1)
                yield epoch_info
                epochs_completed += 1
            except queue.Empty:
                if not training_thread.is_alive():
                    break

        training_thread.join()

        self.evaluate_model()
        self.save_model()
        model_url, scaler_url = self.upload_files_to_api()

        _id = str(uuid.uuid4())
        df = pd.read_csv(self.dataset_url)
        data = df.iloc[int(random.random() *
                           len(df.values.tolist()))].tolist()[0:-1]
        formatted_dat = [f"'{item}'" if isinstance(
            item, str) else str(item) for item in data]

        interference_code = f''' 
import numpy as np
import tensorflow as tf
import joblib
import requests
import io
import pandas as pd

                    
model_url = '{model_url}'    
scaler_url = '{scaler_url}'  
dataset_url = '{self.dataset_url}'  
input_data = [{', '.join(formatted_dat)}] # your input data
                    
def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download url: " + str(response.status_code))

def load_model_from_local(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print("Error loading model from path: " +  str(e))
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

# Download the model and scaler
model_path = model_url.split('/')[-1]
download_file(model_url, model_path)

scaler = load_scaler(scaler_url)

# Load the model from the local file
model = load_model_from_local(model_path)

if model and scaler:
    def preprocess_input(data, scaler, categorical_columns, column_names):
        df = pd.DataFrame([data], columns=column_names)
        df = pd.get_dummies(df, columns=categorical_columns)
        print(scaler.feature_names_in_)
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        data_scaled = scaler.transform(df)
        return data_scaled

    def make_predictions(model, data_scaled):
        predictions = model.predict(data_scaled)
        return predictions

    df = pd.read_csv(dataset_url)
    column_names = df.columns.drop(df.columns[-1])
    categorical_columns = df.select_dtypes(include=['object']).columns

    data_scaled = preprocess_input(input_data, scaler, categorical_columns, column_names)
    predictions = make_predictions(model, data_scaled)

    print(predictions)
else:
    print("Failed to load model or scaler.")

'''
        
        api_code_python = f'''
import requests
import json

url = "https://api.xanderco.in/core/interference/" 

data = {{
    "data": [{', '.join(formatted_dat)}],
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
const url = "https://api.xanderco.in/core/interference/";

const data = {{
    data: [{', '.join(formatted_dat)}],
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
            "modelUrl": model_url if model_url and scaler_url else "",
            "size": (os.path.getsize(self.complete_model_path) / (1024 ** 3) + os.path.getsize(self.complete_scaler_path) / (1024 ** 3)) if model_url and scaler_url else 0,
            "id": _id if model_url and scaler_url else "",
            "helpers": [{"scaler": scaler_url}] if model_url and scaler_url else [],
            "modelArch": self.architecture,
            "hyperparameters": self.hyperparameters,
            "epoch_data": self.epoch_data,
            "task": self.task_type,
            "interferenceCode": interference_code,
            "datasetUrl": self.dataset_url,
            "codes": [
                {"python": api_code_python},
                {"javascript": api_code_js}
            ]
        }
        # os.remove(self.model_file_path)
        # os.remove(self.scaler_file_path)
        yield model_obj if model_url and scaler_url else None