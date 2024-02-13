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
CHECK_POINT = "Salesforce/codet5p-110m-embedding"

checkpoint = CHECK_POINT
device = "cpu"  # for GPU usage or "cpu" for CPU usage

##################### GET THE TOKENIZER AND MODEL FROM HUGGING FACE #####################
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)


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
            
    for i in range(30):
        inputs = tokenizer.encode(program_snippets[i], return_tensors="pt").to(device)
        new_embedding = model(inputs)[0].detach().numpy()
        embedding_vectors.append(new_embedding)

    return embedding_vectors, embedding_locations

def pca_embeddings(df, dimension=2):
    """To reduce the dimensions of the embedding vector we use Principal Component Analysis (PCA).

    :param df: scaled data
    :return: pca result, pca for plotting graph
    """

    pca_transform = PCA(n_components=dimension)
    pca_result = pca_transform.fit_transform(df)
    print('Explained variation per principal component: {}'.format(pca_transform.explained_variance_ratio_))
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(
        np.sum(pca_transform.explained_variance_ratio_)))

    return pca_result, pca_transform

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

    plt.show()

def visualizing_results_3d(pca_result, label, centroids_pca):
    x = pca_result[:, 0]
    y = pca_result[:, 1]
    z = pca_result[:, 2]

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection = "3d")

    ax.scatter3D(x, y, z, c = label)

    ax.scatter3D(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], marker='X', s=200,
                 linewidths=1.5, color='red', edgecolors='black')

    plt.show()
    

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
    if not "codet5p_output" in os.listdir():
        os.mkdir("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/codet5p_output")
    os.chdir("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/codet5p_output")
    for label_type in label_types:
        os.mkdir(os.path.join("/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/codet5p_output", str(label_type)))
    for i in range(len(pca)):
        pca_di = pca[i]
        current_location = os.path.join(data_folder, data_locate[str(pca_di)])
        shutil.copyfile(current_location, "/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/codet5p_output/" + str(kmeans_lables[i]) + "/" + data_locate[str(pca_di)])
        
    return

def pipeline(folder, language):
    """
    The main pipeline
    """
    # Get embeddings with codet5p
    vectors, embedding_locations = load_embeddings(folder, language)
    df = pd.DataFrame(vectors, columns=None)

    # Dimenstion reducing with PCA
    pca, pca_transform = pca_embeddings(df, dimension=3)
    data_locate = {}
    for pca_di, location in zip(pca, embedding_locations):
        data_locate[str(pca_di)] = location 

    # HyperTuning Params for Kmeans
    optimum_num_clusters = kmean_hyper_param_tuning(df)

    # Fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_transform.transform(centroids)
    
    # Clustering files (If we just want to see the plot, we can comment this)
    cluster_files(folder, data_locate, kmeans.labels_, pca)

    # Visualize data
    visualizing_results_3d(pca, kmeans.labels_, centroids_pca)

if __name__ == "__main__":
    # TODO: Replace with the data folder (which contains multiple files)
    # Currently, `language` option won't be used.
    pipeline(folder="/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/data", language="cpp")