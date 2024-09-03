import requests
import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
import uuid
import queue
import threading
import shutil
from tensorflow.keras import regularizers

class ImageModelTrainer:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, userId):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.epoch_info_queue = queue.Queue()
        self.api_url = 'https://api.xanderco.in/core/store/'
        self.data_dir = f"extracted_files{str(uuid.uuid4())}"
        self.img_height = 120
        self.img_width = 120
        self.epoch_data = []
        self.model_path = f'model{str(uuid.uuid4())}.keras'
        self.directory_path = "models"
        self.complete_path = os.path.join(self.directory_path, self.model_path)
        self.userId = userId
        self.download_and_extract_data()
        self.prepare_datasets()
        self.build_model()

    def download_and_extract_data(self):
        temp_zip_path = "data.zip"

        response = requests.get(self.dataset_url)
        with open(temp_zip_path, "wb") as temp_zip_file:
            temp_zip_file.write(response.content)

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        self.data_dir = os.path.join(
            self.data_dir, list(os.listdir(self.data_dir))[0])

        print(self.data_dir)
        print(os.listdir(self.data_dir))

        os.remove(temp_zip_path)
        print(f"Files extracted to: {self.data_dir}")

    def prepare_datasets(self):
        train_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.hyperparameters["batch_size"],
        )

        val_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.hyperparameters["batch_size"],
        )

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.class_names = train_ds.class_names
        print(f"class_names = {self.class_names}")

        self.train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(self):
        self.num_classes = len(self.class_names)

        tf.random.set_seed(123)
        self.model = models.Sequential(
        [
            layers.Rescaling(
                1.0 / 255, input_shape=[self.img_height, self.img_width, 3]),
            layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.35),
            layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.35),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.35),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(self.num_classes),
        ]
    )

        lr = 0.001
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"],
        )

    def train_model(self):
        self.epoch_data = []

        class CustomCallback(callbacks.Callback):
            def __init__(self, outer_instance):
                super().__init__()
                self.outer_instance = outer_instance

            def on_epoch_end(self, epoch, logs=None):
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": logs['loss'],
                    "train_acc": logs['accuracy'],
                    "test_loss": logs['val_loss'],
                    "test_acc": logs['val_accuracy']
                }
                self.outer_instance.epoch_data.append(epoch_info)
                self.outer_instance.epoch_info_queue.put(epoch_info)

        early_stopping = callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        )

        custom_callback = CustomCallback(self)

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.hyperparameters["epochs"],
            callbacks=[early_stopping, custom_callback],
            verbose=0
        )
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
        self.model.save(self.complete_path)

    def upload_files_to_api(self):
        try:
            file = {
                'file': open(self.complete_path, 'rb')
            }
            response_model = requests.post(self.api_url, files=file)
            response_data_model = response_model.json()
            model_url = response_data_model.get('file_url')

            if model_url:
                print(f"Model uploaded successfully. URL: {model_url}")
                return model_url
            else:
                print(
                    f"Failed to upload model. Error: {response_data_model.get('error')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None

    def execute(self):
        training_thread = threading.Thread(target=self.train_model)
        training_thread.start()

        for _ in range(self.hyperparameters["epochs"]):
            try:
                epoch_info = self.epoch_info_queue.get(timeout=300)
                yield epoch_info
            except queue.Empty:
                print(
                    "Timeout waiting for epoch info. Training might have finished early.")
                break

        training_thread.join()

        model_url = self.upload_files_to_api()
        interference_code = f''' 
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# URLs of the uploaded model
model_url = '{model_url}' # URL to the saved model

# Function to download the file from a URL
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

# Prepare the image for prediction
def prepare_image(image_path, img_height, img_width):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Make predictions
def make_predictions(image_path, model, img_height, img_width):
    img_array = prepare_image(image_path, img_height, img_width)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return tf.argmax(score), tf.reduce_max(score)

# Load the model
model = load_model(model_url)

# Define the image path and image dimensions
image_path = 'path_to_your_image.jpg'  # Replace with your image path
img_height = 120
img_width = 120

if model:
    class_idx, class_prob = make_predictions(image_path, model, img_height, img_width)
    print(f"Predicted class index: {{class_idx}}")
    print(f"Class probability: {{class_prob:.4f}}")
else:
    print("Failed to load model.")
        '''
        _id = str(uuid.uuid4())

        api_code_python = f''' 
import requests

url = "https://api.xanderco.in/core/interference/"

data = {{
    'userId': '{self.userId}',
    'modelId': '{_id}',
}}

files = {{
    'image': open('path_to_your_image_file.jpg', 'rb')  # Replace with your image file path
}}

response = requests.post(url, data=data, files=files)

if response.status_code == 200:
    print("Response:")
    print(response.json())
else:
    print(f"Error: {{response.status_code}}")
    print(response.text)

        '''

        api_code_js = f'''
const url = "https://api.xanderco.in/core/interference/";

const data = new FormData();
data.append('userId', '{self.userId}');
data.append('modelId', '{_id}');
data.append('image', document.querySelector('input[type="file"]').files[0]);  // Assuming you have a file input


fetch(url, {{
    method: 'POST',
    body: data
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
        if model_url:
            model_obj = {
                "modelUrl": model_url,
                # size in GB
                "size": os.path.getsize(self.complete_path) / (1024 ** 3),
                "id": _id,
                "modelArch": self.architecture,
                "hyperparameters": self.hyperparameters,
                "epoch_data": self.epoch_data,
                "task": self.task,
                "interferenceCode": interference_code,
                "datasetUrl": self.dataset_url,
                "classnames": self.class_names,
                "codes": [
                    {"python": api_code_python},
                    {"javascript": api_code_js}
                ]
            }
            # os.remove(self.complete_path)
            shutil.rmtree(self.data_dir)
            yield model_obj
        else:
            yield None
