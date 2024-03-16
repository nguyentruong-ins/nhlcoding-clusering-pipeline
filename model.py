import torch.nn as nn
import torch.functional as F
from transformers import AutoModel

# MODEL
class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super(ClassificationHead, self).__init__()
        # self.dense = nn.Linear(256, 256)
        # self.dropout = nn.Dropout(0.0, False)
        # self.out_proj = nn.Linear(256, 12)
        self.dense = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.0, False)
        self.out_proj = nn.Linear(256, 14)

    def forward(self, inputs):
        outputs = self.dense(inputs)

        outputs = F.relu(self.dropout(outputs))
        outputs = F.relu(self.out_proj(outputs))

        return outputs

class CodeT5ClassificationModel(nn.Module):

    def __init__(self, check_point, device):
        super(CodeT5ClassificationModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(check_point, trust_remote_code=True).to(device)
        self.classification_head = ClassificationHead()


    def forward(self, input_ids):
        outputs = self.base_model(input_ids)
        # outputs = self.classification_head(outputs)

        return outputs