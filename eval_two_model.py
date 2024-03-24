# This file is for full pipeline of code embedding and clustering
# Just for evaluate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics.cluster import adjusted_rand_score
from transformers import AutoModel
from model import CodeT5ClassificationModel
from embedding_module import Embedder
from clustering_module import Clusterer
import os

#################################### SET CHECKPOINT ####################################
CHECK_POINT = "Salesforce/codet5p-110m-embedding"
STATE_DICT_FILE = "./pretrained/state_dict_04h_17_03.pt"
device = "cpu"  # for GPU usage or "cpu" for CPU usage

GET_TESTCASE = False

##################### GET THE TOKENIZER AND MODEL FROM HUGGING FACE ####################
base_model = AutoModel.from_pretrained(CHECK_POINT, trust_remote_code=True).to(device)
finetuned_model = CodeT5ClassificationModel(CHECK_POINT, device)
finetuned_model.load_state_dict(torch.load(STATE_DICT_FILE, map_location="cpu"))

def eval():
    """
    The main pipeline
    """
    global finetuned_model, device

    base_ari_scores = []
    finetune_ari_scores = []

    base_embedder = Embedder(CHECK_POINT, base_model, device)
    finetune_embedder = Embedder(CHECK_POINT, finetuned_model, device)
    
    file = "csv/processed_v7_labeled_v3.csv"
    slugs = ["Array_list_7",
             "Array_list_8",
             "Recursion_13",
             "Recursion_14",
             "Sorting_1",
             "Sorting(part_2)_3"]
    
    for slug in slugs:
        clusterer = Clusterer()

        # Clustering process
        base_vectors, _, real_labels, base_testcase_results = base_embedder.load_embeddings_csv(file, slug, eval=True, get_testcase=GET_TESTCASE)
        base_input_vectors = None
        if (len(base_testcase_results) == 0):
            base_input_vectors = base_vectors
        else:
            base_added_testcase_vectors = [np.concatenate([vector, np.array(testcase_result) / 5]) for vector, testcase_result in zip(base_vectors, base_testcase_results)]
            base_input_vectors = base_added_testcase_vectors

        base_df = pd.DataFrame(base_input_vectors, columns=None)
        base_eps, base_min_samples = clusterer.parameters_optimization(base_df, real_labels)
        base_dbscan = clusterer.dbscan_cluster_process(base_df, eps=base_eps, min_samples=base_min_samples)

        base_ari = adjusted_rand_score(real_labels, base_dbscan.labels_)
        base_ari_scores.append(base_ari)

        print("[BASE] Adjuct rand index score with eps={} and min_samples={} of {}: {}".format(base_eps, base_min_samples, slug, base_ari))
        # -------------
        
        finetune_vectors, _, real_labels, finetune_testcase_results = finetune_embedder.load_embeddings_csv(file, slug, eval=True, get_testcase=GET_TESTCASE)
        finetune_input_vectors = None
        if (len(finetune_testcase_results) == 0):
            finetune_input_vectors = finetune_vectors
        else:
            finetune_added_testcase_vectors = [np.concatenate([vector, np.array(testcase_result) / 15]) for vector, testcase_result in zip(finetune_vectors, finetune_testcase_results)]
            finetune_input_vectors = finetune_added_testcase_vectors

        finetune_df = pd.DataFrame(finetune_input_vectors, columns=None)
        finetune_eps, finetune_min_samples = clusterer.parameters_optimization(finetune_df, real_labels)
        finetune_dbscan = clusterer.dbscan_cluster_process(finetune_df, eps=finetune_eps, min_samples=finetune_min_samples)

        # Calculate ARI
        finetune_ari = adjusted_rand_score(real_labels, finetune_dbscan.labels_)
        finetune_ari_scores.append(finetune_ari)
        
        print("[FINETUNE] Adjuct rand index score with eps={} and min_samples={} of {} using {}: {}".format(finetune_eps, finetune_min_samples, slug, STATE_DICT_FILE, finetune_ari))

    # Draw bar chart
    Y_axis = np.arange(len(slugs))
    plt.barh(Y_axis - 0.2, base_ari_scores, 0.4, label = 'Base model') 
    plt.barh(Y_axis + 0.2, finetune_ari_scores, 0.4, label = 'Finetune model') 
    plt.yticks(Y_axis, slugs)
    plt.ylabel("Problems") 
    plt.xlabel("Adjusted Rand Index scores")
    plt.title("Comparison between Base model and Finetune model") 
    plt.legend()

    # Must comment this line in order to save the figure 
    # plt.tight_layout()
    # plt.show()  

    dt_string = datetime.now().strftime("%Hh_%Mm_%Ss_%d_%m")
    plt.tight_layout()
    plt.savefig("eval_two_models_{}.png".format(dt_string))

if __name__ == "__main__":
    eval()