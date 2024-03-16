# This file is for full pipeline of code embedding and clustering
# Just for evaluate

import pandas as pd
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from transformers import AutoModel
from model import CodeT5ClassificationModel
from embedding_module import Embedder
from clustering_module import Clusterer

#################################### SET CHECKPOINT ####################################
CHECK_POINT = "Salesforce/codet5p-110m-embedding"
STATE_DICT_FILE = "./pretrained/state_dict_18h_11_03.pt"
device = "cpu"  # for GPU usage or "cpu" for CPU usage
IS_BASE = False
##################### GET THE TOKENIZER AND MODEL FROM HUGGING FACE ####################

model = None
if IS_BASE:
    model = AutoModel.from_pretrained(CHECK_POINT, trust_remote_code=True).to(device)
else:
    model = CodeT5ClassificationModel(CHECK_POINT, device)
    model.load_state_dict(torch.load(STATE_DICT_FILE, map_location="cpu"))

def pipeline(file, slug):
    """
    The main pipeline
    """
    global model, device

    ########### Embedding module ############
    embedder = Embedder(CHECK_POINT, model, device)
    vectors, program_snippets, _ = embedder.load_embeddings_csv(file, slug)
    df = pd.DataFrame(vectors, columns=None)
 
    ########### Clustering module ###########
    clusterer = Clusterer()

    # Dimension reduce (for visulization purpose)
    pca, pca_transform = clusterer.pca_embeddings(df, dimension=3)
    label_submissions = {}
    for pca_di, snippet in zip(pca, program_snippets):
        label_submissions[str(pca_di)] = snippet 

    # Clustering process
    dbscan = clusterer.cluster_process(df)

    # Clustering files
    clusterer.cluster_files(label_submissions, dbscan.labels_, pca)

    # Visualize data
    clusterer.visualizing_results_3d(pca, dbscan.labels_)

if __name__ == "__main__":
    pipeline("csv/sample_data_v5.csv", "Stack_3")
