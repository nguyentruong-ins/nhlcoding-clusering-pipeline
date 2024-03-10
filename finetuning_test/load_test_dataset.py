from transformers import AutoTokenizer
from datasets import load_dataset

# Codenet dataset
# codenet = load_dataset("systemk/codenet", split="train")
# selected_columns = codenet.select_columns(["code", "status"])

# xCodeEval
xcodeEval = load_dataset("NTU-NLP-sg/xCodeEval", "program_synthesis")





print(xcodeEval.features)