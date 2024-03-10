import pandas as pd
from transformers import AutoModel, AutoTokenizer

#################################### SET CHECKPOINT ####################################
checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cpu"  # for GPU usage or "cpu" for CPU usage

##################### GET THE TOKENIZER AND MODEL FROM HUGGING FACE #####################
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

################## TOKENIZE THE CODE SNIPPET AND PUT IT TO MODEL ##################
# Tokenizer input
with open ("../data/submission_0.cpp", "r") as f:
    snippet = f.read()
    inputs = tokenizer.encode(snippet, return_tensors="pt").to(device)

# Put the tokenizerred input into model in order to generate embedding
embedding = model(inputs)

# print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
# # Dimension of the embedding: 256, with norm=1.0

# Print the embedding
print(embedding)