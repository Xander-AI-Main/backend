import pandas as pd
from ClassificationDL import ClassificationDL
from ClassificationML import ClassificationML
import json

def returnArch (data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if  i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]
        
if __name__ == "__main__":
    dataset_url = "https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/Sonar.csv" # will be sent by user
    hasChanged = True # will be sent by user
    task = "classification" # will be sent by user
    mainType = "DL" # will be sent by user
    archType = "default" # will be sent by user
    arch_data = {} # will be sent by user

    with open ('arch.json', 'r') as f:
        arch_data = json.load(f)

    print(arch_data)

    if task == "classification" and hasChanged == False:
        if mainType == "DL":
            architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
            model_trainer = ClassificationDL(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
        
            executor = model_trainer.execute()
            
            # Iterate through epochs and get epoch information
            for _ in range(hyperparameters['epochs']):
                epoch_info = next(executor)
                print(f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, Train Acc: {epoch_info['train_acc']:.4f}, Test Loss: {epoch_info['test_loss']:.4f}, Test Acc: {epoch_info['test_acc']:.4f}")

            model_obj = next(executor)
            print("Final model object:", model_obj)
        elif mainType == "ML":
            print("In ML")
            architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
            model_trainer = ClassificationML(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
            model_obj = model_trainer.execute()
            print(model_obj)
    if task == "classification" and hasChanged == True:
        if mainType == "DL":
            architecture = [
            {
            "layer": "Dense",
            "neurons": 256,
            "activation": "relu",
            "define_input_shape": "true"
            },
            { "layer": "Dropout", "ratio": 0.1 },
            {
            "layer": "Dense",
            "neurons": 64,
            "activation": "relu",
            "define_input_shape": "false"
            },
            { "layer": "Dropout", "ratio": 0.1 },
            {
            "layer": "Dense",
            "neurons": 32,
            "activation": "relu",
            "define_input_shape": "false"
            },
            { "layer": "Dropout", "ratio": 0.1 },
            {
                "layer": "Dense",
                "neurons": 16,
                "activation": "relu",
                "define_input_shape": "false"
                },
            { "layer": "Dense", "neurons": 1, "define_input_shape": "false", "activation": "sigmoid" }
            ]

            hyperparameters = {"epochs": 50, "batch_size": 32, "validation_size": 0.2}
            model_trainer = ClassificationDL(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
        
            executor = model_trainer.execute()
            
            for _ in range(hyperparameters['epochs']):
                epoch_info = next(executor)
                print(epoch_info)
                # print(f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, Train Acc: {epoch_info['train_acc']:.4f}, Test Loss: {epoch_info['test_loss']:.4f}, Test Acc: {epoch_info['test_acc']:.4f}")

            model_obj = next(executor)
            print("Final model object:", model_obj)
        elif mainType == "ML":
            print("In ML")
            architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)
            model_trainer = ClassificationML(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
            model_obj = model_trainer.execute()
            print(model_obj)

    
   