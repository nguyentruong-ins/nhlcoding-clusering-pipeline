from transformers import AutoModelForSequenceClassification, AutoModel
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

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

        self.base_model = AutoModel.from_pretrained("Salesforce/codet5p-110m-embedding", trust_remote_code=True).to("cpu")
        self.classification_head = ClassificationHead()
        

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids)[0]

        outputs = self.classification_head(outputs)

        return outputs

# model = AutoModel.from_pretrained("Salesforce/codet5p-220m", trust_remote_code=True).to("cpu")
model = AutoModelForSequenceClassification.from_pretrained("Salesforce/codet5p-220m", num_labels=5)
AutoModelForSequenceClassification.from_config
# model = AutoModelForSequenceClassification.from_pretrained("Salesforce/codet5p-220m", trust_remote_code=True, num_labels=7)

# model.base_model.shared.weight.requires_grad = False

# for param in (model.base_model.shared.parameters()):
#     param.requires_grad = False
#     print(1)

# for name, param in model.named_parameters():
#     print(name,param.requires_grad) 
print(model)

# for param in model.parameters():
#     print(param)