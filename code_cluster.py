# This file is for full pipeline of code embedding and clustering
# Just for evaluate

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from transformers import AutoModel
from model import CodeT5ClassificationModel
from embedding_module import Embedder
from clustering_module import Clusterer

#################################### SET CHECKPOINT ####################################
CHECK_POINT = "Salesforce/codet5p-110m-embedding"
STATE_DICT_FILE = "./pretrained/state_dict_01h_16_03.pt"
device = "cpu"  # for GPU usage or "cpu" for CPU usage
IS_BASE = True
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
    vectors, program_snippets, _, testcase_results = embedder.load_embeddings_csv(file, slug)

    input_vectors = None
    if (len(testcase_results) == 0):
        input_vectors = vectors
    else:
        added_testcase_vectors = [np.concatenate([vector, np.array(testcase_result) / 1]) for vector, testcase_result in zip(vectors, testcase_results)]
        input_vectors = added_testcase_vectors

    df = pd.DataFrame(input_vectors, columns=None)

    ########### Clustering module ###########
    clusterer = Clusterer()

    # Dimension reduce (for visulization purpose)
    pca, pca_transform = clusterer.pca_embeddings(df, dimension=3)
    label_submissions = {}
    for pca_di, snippet in zip(pca, program_snippets):
        label_submissions[str(pca_di)] = snippet 

    # Clustering process
    dbscan = clusterer.dbscan_cluster_process(df, eps=0.2, min_samples=3)

    # Clustering files
    clusterer.cluster_files(label_submissions, dbscan.labels_, pca)

    # Visualize data
    clusterer.visualizing_results_3d(pca, dbscan.labels_)

if __name__ == "__main__":
    pipeline("csv/processed_v7_labeled_v3.csv", "Sorting_1")
