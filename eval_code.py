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
GET_TESTCASE = True
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
    vectors, program_snippets, real_labels, testcase_results = embedder.load_embeddings_csv(file, slug, eval=True, get_testcase=GET_TESTCASE)

    input_vectors = None
    if (len(testcase_results) == 0):
        input_vectors = vectors
    else:
        added_testcase_vectors = [np.concatenate([vector, np.array(testcase_result) / 15]) for vector, testcase_result in zip(vectors, testcase_results)]
        input_vectors = added_testcase_vectors


    df = pd.DataFrame(input_vectors, columns=None)

    ########### Clustering module ###########
    clusterer = Clusterer()

    # Parameters optimization
    eps, min_samples = clusterer.parameters_optimization(df, real_labels)


    # Clustering process
    dbscan = clusterer.dbscan_cluster_process(df, eps=eps, min_samples=min_samples)

    ari = adjusted_rand_score(dbscan.labels_, real_labels)
    print("Adjuct rand index score: {}".format(ari))
    print("Chosen eps={}, min_samples={}".format(eps, min_samples))


if __name__ == "__main__":
    pipeline("csv/processed_v7_labeled_v3.csv", "Sorting_1")
