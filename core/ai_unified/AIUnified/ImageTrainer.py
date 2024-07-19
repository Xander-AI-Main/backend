from ImageModelTrainer import ImageModelTrainer


if __name__ == "__main__":
    dataset_url = "https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/train.zip"
    hasChanged = True
    task = "image"
    mainType = "DL"
    archType = "default" 
    architecture = {}    
    hyperparameters = {} 

    import json

    def returnArch (data, task, mainType, archType):
        current_task = data[task]

        for i in current_task:
            if  i["type"] == mainType and i["archType"] == archType:
                return i["architecture"], i["hyperparameters"]

    arch_data = {}
    task = "image"
    mainType='DL'
    archType='default'

    with open ('arch.json', 'r') as f:
        arch_data = json.load(f)

    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)

    trainer = ImageModelTrainer(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
    
    executor = trainer.execute()
    
    for epoch_info in executor:
        if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
            print(f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, Train Acc: {epoch_info['train_acc']:.4f}, Val Loss: {epoch_info['val_loss']:.4f}, Val Acc: {epoch_info['val_acc']:.4f}")
        else:
            model_obj = epoch_info
            break

    if model_obj:
        print(f"Model Object: {model_obj}")
    else:
        print("Failed to train and upload the model.")