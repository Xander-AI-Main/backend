import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import nltk
import os
import requests
import uuid
import pickle
import queue
import threading
from tensorflow.keras.callbacks import Callback


class TextModel:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.api_url = 'https://s3-api-uat.idesign.market/api/upload'
        self.bucket_name = 'idesign-quotation'
        self.epoch_info_queue = queue.Queue()
        self.directory_path = "models"
        self.userId = userId

        self.load_data()
        self.preprocess_data()

    def load_data(self):
        self.df = pd.read_csv(
            self.dataset_url, encoding='latin1', engine='python')

    def preprocess_data(self):
        self.text_columns = []
        self.category_columns = []

        for column in self.df.columns:
            unique_vals = self.df[column].nunique()
            total_vals = len(self.df[column])
            if column.lower() == 'sentiment':
                self.category_columns.append(column)
            elif unique_vals < 0.05 * total_vals:
                self.category_columns.append(column)
            elif self.df[column].dtype == 'object':
                avg_word_length = self.df[column].apply(lambda x: len(
                    str(x).split()) if pd.notnull(x) else 0).mean()
                if avg_word_length > 5:
                    self.text_columns.append(column)

        if not self.text_columns or not self.category_columns:
            raise ValueError(
                "Failed to identify necessary text and category columns.")

        self.category_column = self.category_columns[0]

        print(self.category_column)
        print(self.text_columns[0])

        self.df["text_column"] = self.df[self.text_columns[0]]
        for i, j in enumerate(self.text_columns):
            if i > 0:
                self.df["text_column"] = self.df["text_column"] + \
                    self.df[j].fillna('')

        self.label_encoder = LabelEncoder()
        self.df[self.category_column] = self.label_encoder.fit_transform(
            self.df[self.category_column])
        self.num_classes = len(np.unique(self.df[self.category_column]))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["text_column"], self.df[self.category_column], test_size=0.2, random_state=42
        )

        self.max_num_words = 20000
        self.max_sequence_length = 100

        self.tokenizer = Tokenizer(num_words=self.max_num_words)
        self.tokenizer.fit_on_texts(self.X_train)
        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)

        self.X_train_pad = pad_sequences(
            self.X_train_seq, maxlen=self.max_sequence_length)
        self.X_test_pad = pad_sequences(
            self.X_test_seq, maxlen=self.max_sequence_length)

        self.y_train_cat = to_categorical(self.y_train, self.num_classes)
        self.y_test_cat = to_categorical(self.y_test, self.num_classes)

    def create_model(self):
        self.model = tf.keras.Sequential()
        if self.hasChanged:
            for arch in self.architecture:
                if arch['layer'] == "Embedding":
                    self.model.add(Embedding(
                        input_dim=arch['input_dim'], output_dim=arch['output_dim'], input_length=arch['input_length']))
                elif arch['layer'] == "Bidirectional":
                    self.model.add(Bidirectional(LSTM(
                        units=arch['units'], return_sequences=arch['return_sequences'], dropout=arch['dropout'], recurrent_dropout=arch['recurrent_dropout'])))
                elif arch['layer'] == "LSTM":
                    self.model.add(LSTM(units=arch['units'], dropout=arch['dropout'], recurrent_dropout=arch['recurrent_dropout'], kernel_regularizer=l2(
                        arch['kernel_regularizer']['l2'])))
                elif arch['layer'] == "Dense" and arch.get('define_input_shape', 'false') == "true":
                    self.model.add(Dense(arch['units'], input_shape=(
                        self.X_train_pad.shape[1],), activation=arch['activation'], kernel_regularizer=l2(arch['kernel_regularizer']['l2'])))
                elif arch['layer'] == "Dense" and arch.get('define_input_shape', 'false') == "false":
                    self.model.add(Dense(arch['units'], activation=arch['activation'], kernel_regularizer=l2(
                        arch['kernel_regularizer']['l2'])))
                elif arch['layer'] == "Dropout":
                    self.model.add(Dropout(arch['rate']))
        else:
            self.model.add(Embedding(input_dim=20000,
                           output_dim=100, input_length=100))
            self.model.add(Bidirectional(
                LSTM(units=128, return_sequences=True, dropout=0.45, recurrent_dropout=0.45)))
            self.model.add(Dropout(0.45))
            self.model.add(LSTM(units=64, dropout=0.45,
                           recurrent_dropout=0.45, kernel_regularizer=l2(0.01)))
            self.model.add(Dense(units=self.num_classes,
                           activation='softmax', kernel_regularizer=l2(0.01)))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.batch_size = int(self.hyperparameters["batch_size"])
        self.epochs = int(self.hyperparameters["epochs"])
        self.epoch_data = []

        class CustomCallback(Callback):
            def __init__(self, outer_instance):
                super().__init__()
                self.outer_instance = outer_instance

            def on_epoch_end(self, epoch, logs=None):
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": logs.get('loss'),
                    "train_acc": logs.get('accuracy'),
                    "test_loss": logs.get('val_loss'),
                    "test_acc": logs.get('val_accuracy')
                }
                self.outer_instance.epoch_data.append(epoch_info)
                self.outer_instance.epoch_info_queue.put(epoch_info)

        custom_callback = CustomCallback(self)

        self.history = self.model.fit(
            self.X_train_pad, self.y_train_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.hyperparameters.get("validation_split", 0.2),
            callbacks=[custom_callback],
            verbose=0
        )

    def evaluate_model(self):
        self.loss, self.accuracy = self.model.evaluate(
            self.X_test_pad, self.y_test_cat)
        self.y_pred_prob = self.model.predict(self.X_test_pad)
        self.y_pred = np.argmax(self.y_pred_prob, axis=1)
        print(f"Accuracy: {self.accuracy:.2f}")

    def save_model(self):
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        self.model_path = os.path.join(
            self.directory_path, f'best_model{str(uuid.uuid4())}.h5')
        self.tokenizer_path = os.path.join(
            self.directory_path, f'tokenizer{str(uuid.uuid4())}.pkl')
        self.label_encoder_path = os.path.join(
            self.directory_path, f'label_encoder{str(uuid.uuid4())}.pkl')

        self.model.save(self.model_path)

        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        with open(self.label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def upload_files_to_api(self):
        try:
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.model_path, 'rb')
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
                'files': open(self.tokenizer_path, 'rb')
            }
            response_tokenizer = requests.put(self.api_url, files=files)
            response_data_tokenizer = response_tokenizer.json()
            tokenizer_url = response_data_tokenizer.get(
                'locations', [])[0] if response_tokenizer.status_code == 200 else None

            if tokenizer_url:
                print(f"Tokenizer uploaded successfully. URL: {tokenizer_url}")
            else:
                print(
                    f"Failed to upload tokenizer. Error: {response_data_tokenizer.get('error')}")
                return model_url, None

            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.label_encoder_path, 'rb')
            }
            response_label_encoder = requests.put(self.api_url, files=files)
            response_data_label_encoder = response_label_encoder.json()
            label_encoder_url = response_data_label_encoder.get(
                'locations', [])[0] if response_label_encoder.status_code == 200 else None

            if label_encoder_url:
                print(
                    f"Label encoder uploaded successfully. URL: {label_encoder_url}")
            else:
                print(
                    f"Failed to upload label encoder. Error: {response_data_label_encoder.get('error')}")
                return model_url, tokenizer_url

            return model_url, tokenizer_url, label_encoder_url

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None, None, None

    def execute(self):
        nltk.download('punkt')
        self.create_model()

        training_thread = threading.Thread(target=self.train_model)
        training_thread.start()
        print("---Training started---")

        epochs_completed = 0
        while epochs_completed < self.epochs:
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
        model_url, tokenizer_url, label_encoder_url = self.upload_files_to_api()
        _id = str(uuid.uuid4())

        interference_code = f'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import pickle
import io

# URLs of the uploaded model and helpers
model_url = '{model_url}' # URL to the saved model
tokenizer_url = '{tokenizer_url}' # URL to the saved tokenizer
label_encoder_url = '{label_encoder_url}' # URL to the saved label encoder
input_text = "Your input text here"

# Function to download file from URL
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download {{url}}: {{response.status_code}}")
        return None

# Load the model
def load_model(url):
    model_content = download_file(url)
    if model_content:
        with open(model_url.split("/")[-1], 'wb') as f:
            f.write(model_content)
        model = tf.keras.models.load_model(model_url.split("/")[-1])
        return model
    return None

# Load the tokenizer
def load_tokenizer(url):
    tokenizer_content = download_file(url)
    if tokenizer_content:
        tokenizer = pickle.load(io.BytesIO(tokenizer_content))
        return tokenizer
    return None

# Load the label encoder
def load_label_encoder(url):
    label_encoder_content = download_file(url)
    if label_encoder_content:
        label_encoder = pickle.load(io.BytesIO(label_encoder_content))
        return label_encoder
    return None

# Preprocess the input text
def preprocess_text(text, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# Make predictions
def make_predictions(text, model, tokenizer, label_encoder, max_sequence_length):
    preprocessed_text = preprocess_text(text, tokenizer, max_sequence_length)
    predictions = model.predict(preprocessed_text)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0], predictions[0]

# Load model, tokenizer, and label encoder
model = load_model(model_url)
tokenizer = load_tokenizer(tokenizer_url)
label_encoder = load_label_encoder(label_encoder_url)

# Example text for prediction

if model and tokenizer and label_encoder:
    max_sequence_length = 100  # This should match the max_sequence_length used during training
    predicted_label, prediction_proba = make_predictions(input_text, model, tokenizer, label_encoder, max_sequence_length)
    print(f"Predicted label: {{predicted_label}}")
    print(f"Prediction probabilities: {{prediction_proba}}")
else:
    print("Failed to load model, tokenizer, or label encoder.")
        '''

        api_code_python = f'''
import requests
import json

url = "https://api.xanderco.in/core/interference/" 

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
const url = "https://api.xanderco.in/core/interference/";

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

        if model_url and tokenizer_url and label_encoder_url:
            model_obj = {
                "modelUrl": model_url,
                "size": os.path.getsize(self.model_path) / (1024 ** 3) + os.path.getsize(self.label_encoder_path) / (1024 ** 3) + os.path.getsize(self.tokenizer_path) / (1024 ** 3),
                "id": _id,
                "helpers": [{"tokenizer": tokenizer_url}, {"label_encoder": label_encoder_url}],
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
            # os.remove(self.tokenizer_path)
            # os.remove(self.label_encoder_path)
            yield model_obj
        else:
            yield None
