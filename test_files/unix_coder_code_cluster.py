# This file is for full pipeline of code embedding and clustering
# Just for evaluate

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import shutil
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

#################################### SET CHECKPOINT ####################################
import torch
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base-nine")
model.to(device)

def load_embeddings(folder, language):
    """
    To get the embeddings from specific folder with specific language

    :param folder: the data's folder path (should store only the submissions that belong to 1 problem with 1 language)
    :return: embedding_vectors - The embedding vectors that are created; embedding_locations - The location of the file which is embedded respectively
    """
    program_snippets = []
    embedding_vectors = []
    embedding_locations = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r") as f:
            new_snippet = f.read()
            program_snippets.append(new_snippet)
            embedding_locations.append(file)

    # Can this model embedding batch?
    # 20 is for testing
    for i in range(30):
        token_ids = model.tokenize([program_snippets[i]], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(token_ids).to("cpu")
        tokens_embedding,program_embeddings = model(source_ids)
        new_embedding = program_embeddings[0].detach().numpy()
        embedding_vectors.append(new_embedding)

    return embedding_vectors, embedding_locations

def pca_embeddings(df):
    """To reduce the dimensions of the wine dataset we use Principal Component Analysis (PCA).
    Here we reduce it from 13 dimensions to 2.

    :param df: scaled data
    :return: pca result, pca for plotting graph
    """

    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(df)
    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))

    # Results from pca.components_
    dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=df.columns, index=['PC_1', 'PC_2'])
    print('\n\n', dataset_pca)
    
    print("\n*************** Most important features *************************")
    print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
    print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
    print("\n******************************************************************")

    return pca_2_result, pca_2

def visualizing_results(pca_result, label, centroids_pca):
    """ Visualizing the clusters

    :param pca_result: PCA applied data
    :param label: K Means labels
    :param centroids_pca: PCA format K Means centroids
    """
    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    # plt.scatter(x, y, alpha=0.5, s=200)  # plot different colors per cluster
    plt.scatter(x, y, c=label, alpha=0.5, s=200)  # plot different colors per cluster
    plt.title('Submission clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black")

    # plt.show()
    plt.savefig("./unix_coder.png")

def kmean_hyper_param_tuning(data):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    # parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
    parameters = [3]

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans()     # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()

    return best_grid['n_clusters']

def cluster_files(data_folder, data_locate, kmeans_lables, pca):
    """
    Help us cluster the submissions after vectorizing into right folder

    :param data_folder: the dataset folder
    :param data_locate: The file's location after embedding
    :param kmeans_lables: The kmeans label
    :param pca: The pca result

    """
    label_types = collections.Counter(kmeans_lables).keys()
    os.chdir("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering")
    if not "unixcoder_output" in os.listdir():
        os.mkdir("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/unixcoder_output")
    os.chdir("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/unixcoder_output")
    for label_type in label_types:
        os.mkdir(os.path.join("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/unixcoder_output", str(label_type)))
    for i in range(len(pca)):
        pca_di = pca[i]
        current_location = os.path.join(data_folder, data_locate[str(pca_di)])
        shutil.copyfile(current_location, "/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/unixcoder_output/" + str(kmeans_lables[i]) + "/" + data_locate[str(pca_di)])
        
    return

def pipeline(folder, language):
    """
    The main pipeline
    """
    # Get embeddings with codet5p
    vectors, embedding_locations = load_embeddings(folder, language)
    df = pd.DataFrame(vectors, columns=None)

    # print(vectors, embedding_locations)

    # Dimenstion reducing with PCA
    pca, pca_2 = pca_embeddings(df)
    data_locate = {}
    for pca_di,location in zip(pca, embedding_locations):
        data_locate[str(pca_di)] = location 

    # HyperTuning Params for Kmeans
    optimum_num_clusters = kmean_hyper_param_tuning(df)

    # fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_2.transform(centroids)
    
    # Clustering files
    cluster_files(folder, data_locate, kmeans.labels_, pca)

    # Visualize data
    visualizing_results(pca, kmeans.labels_, centroids_pca)

if __name__ == "__main__":
    pipeline("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/data", "cpp")