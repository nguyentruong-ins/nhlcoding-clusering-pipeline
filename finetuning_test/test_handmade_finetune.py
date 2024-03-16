# IMPORT LIBRARIES
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm.auto import tqdm

# CONSTANTS
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
DATASET = "systemk/codenet"

# DATA: Just for testing
# TODO: Update to our dataset 
def get_train_data(dataset):
    datasets = load_dataset(DATASET, split="train[10:20]")
    datasets = datasets.select_columns(["code", "status"])
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    def preprocess_function(examples):
        return tokenizer(examples["code"], max_length=200, padding="max_length", truncation=True)
    train_data = datasets.map(
        preprocess_function,
        batched=True
    )
    
    train_data = train_data.rename_columns({'status': 'labels'})
    train_data = train_data.remove_columns(['code'])
    train_data.set_format("torch")

    return train_data

# MODEL
class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.0, False)
        self.out_proj = nn.Linear(256, 7)

    def forward(self, inputs):
        outputs = self.dense(inputs)

        outputs = F.relu(self.dropout(outputs))
        outputs = F.relu(self.out_proj(outputs))

        return outputs

class CodeT5ClassificationModel(nn.Module):
    def __init__(self):
        super(CodeT5ClassificationModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).to("cpu")
        self.classification_head = ClassificationHead()
        

    def forward(self, input_ids):
        outputs = self.base_model(input_ids)[0]

        outputs = self.classification_head(outputs)

        return outputs

# FINETUNING
train_data = get_train_data(DATASET)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1)

model = CodeT5ClassificationModel()

print(model)

# optimizer = AdamW(model.parameters(), lr=5e-5)

# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# progress_bar = tqdm(range(num_training_steps))

# model.train()
# criterion = nn.CrossEntropyLoss()
# y = torch.eye(7)
# for epoch in range(num_epochs):

#     # To calculate average loss and print it
#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         inputs = data['input_ids']
#         labels = y[data['labels'][0]]
        
#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         progress_bar.update(1)

# if "save_model" not in os.listdir():
#     os.mkdir("save_model")

# torch.save(model.state_dict(), "./save_model")

    # for batch in train_dataloader:
    #     print(batch)
    #     break
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     outputs = model(**batch)
    #     print(outputs)
    #     break
    #     loss = outputs.loss
    #     loss.backward()

    #     optimizer.step()
    #     lr_scheduler.step()
    #     optimizer.zero_grad()
    #     progress_bar.update(1)