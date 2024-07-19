from TextModel import TextModel
import json

def returnArch (data, task, mainType, archType):
    current_task = data[task]

    for i in current_task:
        if  i["type"] == mainType and i["archType"] == archType:
            return i["architecture"], i["hyperparameters"]

arch_data = {}
task = "text"
mainType='topic classification'
archType='default'

with open ('arch.json', 'r') as f:
    arch_data = json.load(f)

architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)

model = TextModel(
    dataset_url='bbc-text.csv',
    hasChanged=False,
    task='text',
    mainType='topic classification',
    archType='default',
    architecture=architecture,
    hyperparameters=hyperparameters
)

executor = model.execute()
for epoch_info in executor:
    if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
        print(f"Epoch {epoch_info['epoch']}: Train Loss: {epoch_info['train_loss']:.4f}, "
              f"Train Acc: {epoch_info['train_acc']:.4f}, Val Loss: {epoch_info['val_loss']:.4f}, "
              f"Val Acc: {epoch_info['val_acc']:.4f}")
    else:
        print("Final model object:", epoch_info)
        break