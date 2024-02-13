# # Load model directly
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
# model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine")

# ################## TOKENIZE THE CODE SNIPPET AND PUT IT TO MODEL ##################
# # Tokenizer input
# with open ("./data/1.py", "r") as f:
#     snippet = f.read()
#     inputs = tokenizer.tokenize([snippet]).to("cpu")

# # Put the tokenizerred input into model in order to generate embedding
# embedding = model(inputs)[0]

# # print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
# # # Dimension of the embedding: 256, with norm=1.0

# # Print the embedding
# print(embedding.size)

import torch
from unixcoder import UniXcoder

snippet = None
with open ("./data/1.py", "r") as f:
    snippet = f.read()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base-nine")
model.to(device)

# Encode maximum function
func = snippet
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,max_func_embedding = model(source_ids)

print(tokens_embeddings)
print(max_func_embedding)