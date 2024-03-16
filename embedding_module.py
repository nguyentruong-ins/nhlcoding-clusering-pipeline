import pandas as pd
from cleaner import code_preprocess 
from transformers import AutoTokenizer

class Embedder():
    # The model should be loadded with trained state dict
    def __init__(self, checkpoint, model, device) -> None:
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.device = device


    def load_embeddings_csv(self, file_path, slug, eval=False):
        """
        To get the embeddings from specific file

        :param folder: the file path
        :return: embedding_vectors - The embedding vectors that are created; embedding_locations - The location of the file which is embedded respectively
        """

        df = pd.read_csv(file_path)
        df = df[(df['problem_slug'] == slug)]
        program_snippets = df['code'].to_list()
        remove_comment_snippets = [code_preprocess(snippet) for snippet in program_snippets]
        real_labels = []
        if eval:
            real_labels = df['labels'].to_list()

        embedding_vectors = []

        for snippet in remove_comment_snippets:
            inputs = self.tokenizer.encode(snippet, return_tensors="pt").to(self.device)
            # inputs = tokenizer.encode(remove_comment_snippet, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(device)
            new_embedding = self.model(inputs)[0].detach().numpy()
            embedding_vectors.append(new_embedding)

        return embedding_vectors, program_snippets, real_labels