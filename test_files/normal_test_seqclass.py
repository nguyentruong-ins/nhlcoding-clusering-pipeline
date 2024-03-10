from transformers import AutoModelForSequenceClassification, AutoModel
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.0, False)
        self.out_proj = nn.Linear(256, 5)

    def forward(self, inputs):
        outputs = self.dense(inputs)

        outputs = F.relu(self.dropout(outputs))
        outputs = F.relu(self.out_proj(outputs))

        return outputs

class CodeT5ClassificationModel(nn.Module):
    def __init__(self):
        super(CodeT5ClassificationModel, self).__init__()

        self.base_model = AutoModel.from_pretrained("Salesforce/codet5p-220m", trust_remote_code=True).to("cpu")
        self.classification_head = ClassificationHead()
        

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids)[0]

        outputs = self.classification_head(outputs)

        return outputs

model = CodeT5ClassificationModel()

# for name, param in model.named_parameters():
#     print(name,param.requires_grad) 
print(model)