import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Add, Activation, Input
from tensorflow.keras.callbacks import Callback
import joblib
import requests
import os
import uuid
import queue
import threading
import random
import numpy as np
from tensorflow.keras.regularizers import l2
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

class ClassificationDL:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.api_url = 'https://api.xanderco.in/core/store/'
        self.model_path = f'bestmodel{str(uuid.uuid4())}.h5'
        self.scaler_path = f'scaler{str(uuid.uuid4())}.pkl'
        self.label_encoder_path = f'label_encoder{str(uuid.uuid4())}.pkl'
        self.directory_path = "models"
        self.complete_model_path = os.path.join(self.directory_path, self.model_path)
        self.complete_scaler_path = os.path.join(self.directory_path, self.scaler_path)
        self.complete_label_encoder_path = os.path.join(self.directory_path, self.label_encoder_path)
        self.userId = userId
        self.coff = 0

        self.load_data()
        self.preprocess_data()

    def load_data(self):
        self.df = pd.read_csv(self.dataset_url)
        print(self.df)
        self.df = self.df.dropna()
        self.df = self.df.iloc[:25000]
        columns_to_drop = [col for col in self.df.columns if 'id' in col.lower()]
        self.df = self.df.drop(columns=columns_to_drop)
        
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.label_encoders = {}
        for column in self.X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column])
            self.label_encoders[column] = le

        if self.y.dtype == object:
            self.y = LabelEncoder().fit_transform(self.y)

        # correlation_matrix = self.X.corr().abs()
        # upper_triangle = correlation_matrix.where(
        #     pd.np.triu(pd.np.ones(correlation_matrix.shape), k=1).astype(pd.np.bool_)
        # )
        # self.coff = upper_triangle.stack().mean()

        # tree_clf = DecisionTreeClassifier(max_depth=3)
        # tree_clf.fit(self.X, self.y)
        
        # feature_importances = tree_clf.feature_importances_
        
        # mutual_info_scores = mutual_info_classif(self.X, self.y)
        
        # features_to_drop = [i for i in range(len(feature_importances)) 
        #                     if feature_importances[i] == 0 or mutual_info_scores[i] < 0.01]

        # self.X = self.X.drop(self.X.columns[features_to_drop], axis=1)
        # print(self.X)
        # print(len(list(self.df.values)))
        # print("Final Features:", self.X.columns)


    def preprocess_data(self):
        self.scaler = StandardScaler()
        self.X_standardized = self.scaler.fit_transform(self.X)
        self.X = self.X_standardized
        self.y = self.y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        
    def residual_block(self, x, units, dropout_rate=0.3):
        """A residual block with two Dense layers and a skip connection."""
        shortcut = x
        
        # Adjust shortcut to match the new shape
        if int(shortcut.shape[-1]) != units:
            shortcut = Dense(units, activation=None)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)
        
        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        return x
    
    def create_model(self):
        unique_values = np.unique(self.y)
        isBinary = len(unique_values) <= 2
        num_classes = len(unique_values)

        print(self.X_train.shape[1])
        if self.task == "classification" and self.hasChanged:
            for arch in self.architecture:
                if arch['layer'] == "Dense" and arch['define_input_shape'] == "true":
                    self.model.add(Dense(arch['neurons'], input_shape=(self.X_train.shape[1],), activation=arch['activation']))
                elif arch['layer'] == "Dense" and arch['define_input_shape'] == "false":
                    self.model.add(Dense(arch['neurons'], activation=arch['activation']))
                elif arch['layer'] == "Dropout":
                    self.model.add(Dropout(arch['ratio']))
        else:
            if len(list(self.df.values)) <= 2000:
                self.model = tf.keras.Sequential()
                self.model.add(Dense(512, kernel_regularizer=l2(0.001), input_shape=(self.X_train.shape[1],), activation="relu"))
                self.model.add(Dropout(0.3))
                self.model.add(BatchNormalization())

                self.model.add(Dense(256, kernel_regularizer=l2(0.001), activation="relu"))
                self.model.add(Dropout(0.3))
                self.model.add(BatchNormalization())

                self.model.add(Dense(128, kernel_regularizer=l2(0.001), activation="relu"))
                self.model.add(Dropout(0.25))
                # self.model.add(BatchNormalization())

                self.model.add(Dense(64, kernel_regularizer=l2(0.001), activation="relu"))
                self.model.add(Dropout(0.15))
                # self.model.add(BatchNormalization())

                self.model.add(Dense(32, kernel_regularizer=l2(0.001), activation="relu"))
                # self.model.add(BatchNormalization())
                # self.model.add(Dropout(0.15))

                if isBinary:
                    self.model.add(Dense(1, activation='sigmoid'))
                    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    self.model.add(Dense(num_classes, activation='softmax'))
                    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            elif len(list(self.df.values)) <= 5000:
                self.model = tf.keras.Sequential()
                self.model.add(Dense(512, kernel_regularizer=l2(0.01), input_shape=(self.X_train.shape[1],), activation="relu"))
                self.model.add(Dropout(0.3))
                self.model.add(BatchNormalization())

                self.model.add(Dense(256, kernel_regularizer=l2(0.01), activation="relu"))
                self.model.add(Dropout(0.3))
                self.model.add(BatchNormalization())

                self.model.add(Dense(128, kernel_regularizer=l2(0.01), activation="relu"))
                self.model.add(Dropout(0.25))
                # self.model.add(BatchNormalization())

                self.model.add(Dense(64, kernel_regularizer=l2(0.01), activation="relu"))
                self.model.add(Dropout(0.15))
                # self.model.add(BatchNormalization())

                self.model.add(Dense(32, kernel_regularizer=l2(0.01), activation="relu"))
                # self.model.add(BatchNormalization())
                # self.model.add(Dropout(0.15))

                if isBinary:
                    self.model.add(Dense(1, activation='sigmoid'))
                    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    self.model.add(Dense(num_classes, activation='softmax'))
                    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            else:
                inputs = Input(shape=(self.X_train.shape[1],))
                x = inputs
                
                x = Dense(512, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)

                x = self.residual_block(x, 512, dropout_rate=0.3)
                x = self.residual_block(x, 256, dropout_rate=0.3)
                x = self.residual_block(x, 128, dropout_rate=0.3)
                x = self.residual_block(x, 64, dropout_rate=0.2)

                x = Dense(32, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)

                if isBinary:
                    outputs = Dense(1, activation='sigmoid')(x)
                    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    outputs = Dense(num_classes, activation='softmax')(x)
                    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.epoch_data = []
        self.current_val_acc = 0
        class CustomCallback(Callback):
            def __init__(self, outer_instance, validation_data):
                super().__init__()
                self.outer_instance = outer_instance
                self.validation_data = validation_data

            def on_epoch_end(self, epoch, logs=None):
                test_loss, test_acc = self.model.evaluate(
                    self.validation_data[0], self.validation_data[1], verbose=0)
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

                if test_acc > self.outer_instance.current_val_acc:
                    self.outer_instance.save_model()
                    self.outer_instance.current_val_acc = test_acc
                    print(f"New best model saved with validation accuracy: {test_acc}")

        self.epoch_info_queue = queue.Queue()
        custom_callback = CustomCallback(
            self, validation_data=(self.X_test, self.y_test))

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

    def save_model(self):
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
            model_path = os.path.join(self.directory_path, self.model_path)
            scaler_path = os.path.join(self.directory_path, self.scaler_path)
            label_encoder_path = os.path.join(self.directory_path, self.label_encoder_path)
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoders, label_encoder_path)
        else:
            model_path = os.path.join(self.directory_path, self.model_path)
            scaler_path = os.path.join(self.directory_path, self.scaler_path)
            label_encoder_path = os.path.join(self.directory_path, self.label_encoder_path)
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoders, label_encoder_path)

    def upload_files_to_api(self):
        try:
            file = {
                'file': open(self.complete_model_path, 'rb')
            }
            response_model = requests.post(self.api_url, files=file)
            response_data_model = response_model.json()
            model_url = response_data_model.get('file_url')

            if model_url:
                print(f"Model uploaded successfully. URL: {model_url}")
            else:
                print(
                    f"Failed to upload model. Error: {response_data_model.get('error')}")
                return None, None

            file = {
                'file': open(self.complete_scaler_path, 'rb')
            }

            response_scaler = requests.post(self.api_url, files=file)
            response_data_scaler = response_scaler.json()
            scaler_url = response_data_scaler.get('file_url')

            if scaler_url:
                print(f"Scaler uploaded successfully. URL: {scaler_url}")
            else:
                print(
                    f"Failed to upload scaler. Error: {response_data_scaler.get('error')}")
                return model_url, None
            
            file = {
                'file': open(self.complete_label_encoder_path, 'rb')
            }

            response_label = requests.post(self.api_url, files=file)
            response_data_label = response_label.json()
            label_url = response_data_label.get('file_url')

            if label_url:
                print(f"label uploaded successfully. URL: {label_url}")
            else:
                print(
                    f"Failed to upload label. Error: {response_data_label.get('error')}")
            
            print(model_url, scaler_url, label_url)
            return model_url, scaler_url, label_url

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

        # self.evaluate_model()
        # model_url = ""
        # scaler_url = ""
        # label_url = ""
        
        model_url, scaler_url, label_url = self.upload_files_to_api()
        
        _id = str(uuid.uuid4())

        df = pd.read_csv(self.dataset_url)
        data = df.iloc[int(random.random() *
                           len(df.values.tolist()))].tolist()[0:-1]
        formatted_dat = [f"'{item}'" if isinstance(
            item, str) else str(item) for item in data]
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
input_data = [{', '.join(formatted_dat)}]  # your input data

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
        print(f"Error loading model: {{e}}")
        return None

def load_scaler(url):
    response = requests.get(url)
    if response.status_code == 200:
        scaler_content = response.content
        try:
            scaler = joblib.load(io.BytesIO(scaler_content))
            return scaler
        except Exception as e:
            print(f"Error loading scaler: {{e}}")
            return None
    else:
        print(f"Failed to download scaler: {{response.status_code}}")
        return None

def preprocess_input(data, scaler, categorical_columns, column_names):
    df = pd.DataFrame([data], columns=column_names)
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    data_scaled = scaler.transform(df)
    return data_scaled

def make_predictions(model, data_scaled):
    predictions_proba = model.predict(data_scaled)
    if predictions_proba.shape[1] == 1:  # Binary classification
        predictions = (predictions_proba > 0.5).astype(int)
    else:  # Multi-class classification
        predictions = np.argmax(predictions_proba, axis=1)
    return predictions, predictions_proba

if __name__ == "__main__":
    model_path = model_url.split('/')[-1]
    download_file(model_url, model_path)
    scaler = load_scaler(scaler_url)

    model = load_model_from_local(model_path)

    if model and scaler:
        df = pd.read_csv(dataset_url)
        column_names = df.columns.drop(df.columns[-1]).tolist()
        categorical_columns = df.drop(columns=df.columns[-1]).select_dtypes(include=['object']).columns.tolist()

        data_scaled = preprocess_input(input_data, scaler, categorical_columns, column_names)

        predictions, predictions_proba = make_predictions(model, data_scaled)

        if predictions_proba.shape[1] == 1:  # Binary classification
            print("Predicted class:", predictions[0][0])
            print("Prediction probability:", predictions_proba[0][0])
        else:  # Multi-class classification
            print("Predicted class:", predictions[0])
            print("Prediction probabilities:", predictions_proba[0])
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
            "size": os.path.getsize(self.complete_model_path) / (1024 ** 3) if model_url and scaler_url else 0,
            "id": _id if model_url and scaler_url else "",
            "helpers": [{"scaler": scaler_url}, {"label_encoders": label_url}] if model_url and scaler_url else [],
            "modelArch": self.architecture,
            "hyperparameters": self.hyperparameters,
            "epoch_data": self.epoch_data,
            "task": self.task,
            "interferenceCode": interference_code,
            "datasetUrl": self.dataset_url,
            "codes": [
                {"python": api_code_python},
                {"javascript": api_code_js}
            ]
        }
        # os.remove(self.model_path)
        # os.remove(self.scaler_path)
        yield model_obj if model_url and scaler_url else None
