import pandas as pd
from RegressionML import RegressionML
from RegressionDL import RegressionDL
import json

def returnArch(data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]

if __name__ == "__main__":
    dataset_url = "test.csv"  # will be sent by user
    hasChanged = False  # will be sent by user
    task = "regression"  # will be sent by user
    mainType = "DL"  # will be sent by user
    archType = "default"  # will be sent by user
    arch_data = {}  # will be sent by user if hasChanged = true
    
    with open('arch.json', 'r') as f:
        arch_data = json.load(f)

    # print(arch_data)

    if task == "regression" and hasChanged == False:
        if mainType == "DL":
            architecture, hyperparameters = returnArch(
                arch_data, task, mainType, archType)
            model_trainer = RegressionDL(
                dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, "9")
            print("executing")
            executor = model_trainer.execute()
            
            for epoch_info in executor:
                if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                    print(f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, Test Loss: {epoch_info['test_loss']:.4f}")
                else:
                    print("Final model object:", epoch_info)
                    break

        elif mainType == "ML":
            print("In ML")
            architecture, hyperparameters = returnArch(
                arch_data, task, mainType, archType)
            model_trainer = RegressionML(
                dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, "9")
            model_obj = model_trainer.execute()
            print(model_obj)
    if task == "regression" and hasChanged == True:
        if mainType == "DL":
            architecture = [
                {
                    "layer": "Dense",
                    "neurons": 128,
                    "activation": "relu",
                    "define_input_shape": "true"
                },
                {"layer": "Dropout", "ratio": 0.1},
                {
                    "layer": "Dense",
                    "neurons": 64,
                    "activation": "relu",
                    "define_input_shape": "false"
                },
                {"layer": "Dropout", "ratio": 0.1},
                {
                    "layer": "Dense",
                    "neurons": 32,
                    "activation": "relu",
                    "define_input_shape": "false"
                },
                {"layer": "Dense", "neurons": 1, "define_input_shape": "false"}
            ]

            hyperparameters = {"epochs": 1,
                               "batch_size": 32, "validation_size": 0.2}
            model_trainer = RegressionDL(
                dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, "9")
            model_obj = model_trainer.execute()
            print(model_obj)
        elif mainType == "ML":
            print("In ML")
            architecture, hyperparameters = returnArch(
                arch_data, task, mainType, archType)
            model_trainer = RegressionML(
                dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters, "9")
            model_obj = model_trainer.execute()
            print(model_obj)
