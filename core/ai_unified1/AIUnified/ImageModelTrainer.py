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

class ImageModelTrainer:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.epoch_info_queue = queue.Queue()
        self.api_url = 'https://s3-api-uat.idesign.market/api/upload'
        self.bucket_name = 'idesign-quotation'
        self.data_dir = "extracted_files"
        self.img_height = 120
        self.img_width = 120

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
        
        self.data_dir = os.path.join(self.data_dir, list(os.listdir(self.data_dir))[0])

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
                layers.Rescaling(1.0 / 255, input_shape=[self.img_height, self.img_width, 3]),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.num_classes),
            ]
        )

        lr = 0.001
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_model(self):
        class CustomCallback(callbacks.Callback):
            def __init__(self, outer_instance):
                super().__init__()
                self.outer_instance = outer_instance

            def on_epoch_end(self, epoch, logs=None):
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": logs['loss'],
                    "train_acc": logs['accuracy'],
                    "val_loss": logs['val_loss'],
                    "val_acc": logs['val_accuracy']
                }
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

        self.model.save("model.h5")

    def upload_files_to_api(self):
        try:
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open("model.h5", 'rb')
            }
            response_model = requests.put(self.api_url, files=files)
            response_data_model = response_model.json()
            model_url = response_data_model.get('locations', [])[0] if response_model.status_code == 200 else None
            
            if model_url:
                print(f"Model uploaded successfully. URL: {model_url}")
                return model_url
            else: 
                print(f"Failed to upload model. Error: {response_data_model.get('error')}")
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
                print("Timeout waiting for epoch info. Training might have finished early.")
                break

        training_thread.join()

        model_url = self.upload_files_to_api()
        
        if model_url:
            _id = str(uuid.uuid4())
            model_obj = {
                "modelUrl": model_url,
                "size": os.path.getsize("model.h5") / (1024 ** 3),  # size in GB
                "id": _id,
                "modelArch": self.architecture,
                "hyperparameters": self.hyperparameters
            }
            yield model_obj
        else:
            yield None
