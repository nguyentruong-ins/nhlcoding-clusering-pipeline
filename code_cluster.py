# This file is for full pipeline of code embedding and clustering
# Just for evaluate

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans, DBSCAN
from transformers import AutoModel, AutoTokenizer

#################################### SET CHECKPOINT ####################################
CHECK_POINT = "Salesforce/codet5p-110m-embedding"
STATE_DICT_FILE = "./pretrained/state_dict_06h_03_03.pt"

checkpoint = CHECK_POINT
device = "cpu"  # for GPU usage or "cpu" for CPU usage


# MODEL
class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super(ClassificationHead, self).__init__()
        # self.dense = nn.Linear(256, 256)
        # self.dropout = nn.Dropout(0.0, False)
        # self.out_proj = nn.Linear(256, 12)
        self.dense1 = nn.Linear(256, 256)
        self.dense2 = nn.Linear(256, 128)
        self.out_proj = nn.Linear(128, 7)

    def forward(self, inputs):
        outputs = self.dense(inputs)

        outputs = F.relu(self.dropout(outputs))
        outputs = F.relu(self.out_proj(outputs))

        return outputs

class CodeT5ClassificationModel(nn.Module):
    def __init__(self):
        super(CodeT5ClassificationModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(CHECK_POINT, trust_remote_code=True).to(device)
        self.classification_head = ClassificationHead()


    def forward(self, input_ids):
        outputs = self.base_model(input_ids)
        # outputs = self.classification_head(outputs)

        return outputs

def comment_remover(code):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, code)

def format_code(code, style='google'):
    # Run clang-format with the specified style and capture the formatted code
    formatted_code = subprocess.run(
        ['clang-format', '-style=' + style], 
        input=code.encode('utf-8'), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    # Return the formatted code as a string
    return formatted_code.stdout.decode('utf-8')

def code_preprocess(code):
    comment_free_code = comment_remover(code)
    formatted_code = format_code(comment_free_code)

    return formatted_code

##################### GET THE TOKENIZER AND MODEL FROM HUGGING FACE #####################
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# model = CodeT5ClassificationModel()
# model.load_state_dict(torch.load(STATE_DICT_FILE, map_location="cpu"))

def load_embeddings(folder):
    """
    To get the embeddings from specific folder with specific language

    :param folder: the data's folder path (should store only the submissions that belong to 1 problem with 1 language)
    :return: embedding_vectors - The embedding vectors that are created; embedding_locations - The location of the file which is embedded respectively
    """
    program_snippets = []
    embedding_vectors = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r") as f:
            new_snippet = f.read()
            program_snippets.append(new_snippet)
            
    for i in range(30):
        remove_commnet_snippet = code_preprocess(program_snippets[i])
        inputs = tokenizer.encode(remove_commnet_snippet, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(device)
        # inputs = tokenizer(program_snippets[i], max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(device)
        new_embedding = model(inputs)[0].detach().numpy()
        embedding_vectors.append(new_embedding)

    return embedding_vectors, program_snippets

def load_embeddings_csv(file_path, problem_name):
    """
    To get the embeddings from specific file

    :param folder: the file path
    :return: embedding_vectors - The embedding vectors that are created; embedding_locations - The location of the file which is embedded respectively
    """

    df = pd.read_csv(file_path)
    filter_df = df[(df['problem_slug'] == problem_name)]

    program_snippets = filter_df['code'].to_list()
    embedding_vectors = []

    for snippet in program_snippets:
        remove_comment_snippet = code_preprocess(snippet)
        inputs = tokenizer.encode(remove_comment_snippet, return_tensors="pt").to(device)
        # inputs = tokenizer.encode(remove_comment_snippet, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(device)
        new_embedding = model(inputs)[0].detach().numpy()
        embedding_vectors.append(new_embedding)
    
    return embedding_vectors, program_snippets

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

def visualizing_results_3d(pca_result, label, problem_name = None, centroids_pca = None):
    x = pca_result[:, 0]
    y = pca_result[:, 1]
    z = pca_result[:, 2]

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection = "3d")

    scatter = ax.scatter3D(x, y, z, c = label)
    # Adding labels for data points
    legend = ax.legend(*scatter.legend_elements(),
                       title="Legend", loc='upper left')
    ax.add_artist(legend)
    
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], '%s' % (str(i)), size=13, zorder=1)

    # ax.scatter3D(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], marker='X', s=200,
    #              linewidths=1.5, color='red', edgecolors='black')

    ax.legend()

    plt.show()
    # plt.savefig("{}.png".format(problem_name))
    

def kmean_hyper_param_tuning(data):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5]
    # parameters = [3]

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

def cluster_files(data_locate, kmeans_lables, pca):
    """
    Help us cluster the submissions after vectorizing into right folder

    :param data_folder: the dataset folder
    :param data_locate: The file's location after embedding
    :param kmeans_lables: The kmeans label
    :param pca: The pca result

    """
    label_types = collections.Counter(kmeans_lables).keys()

    if "codet5p_output" in os.listdir():
        shutil.rmtree("codet5p_output")

    os.mkdir("codet5p_output")

    for label_type in label_types:
        os.mkdir(os.path.join("codet5p_output", str(label_type)))

    for i in range(len(pca)):
        pca_di = pca[i]
        snippet = data_locate[str(pca_di)]
        file_path = "codet5p_output/{}/submission_{}.cpp".format(str(kmeans_lables[i]), i)

        with open(file_path, "w") as f:
            f.write(snippet)
            f.close()

    return

def pipeline(folder, problem_name):
    """
    The main pipeline
    """
    # Get embeddings with codet5p
    # vectors, program_snippets = load_embeddings(folder)
    vectors, program_snippets = load_embeddings_csv("csv/sample_data_v5.csv", problem_name)
    df = pd.DataFrame(vectors, columns=None)

    # Dimenstion reducing with PCA
    pca, pca_transform = pca_embeddings(df, dimension=3)
    label_submissions = {}
    for pca_di, snippet in zip(pca, program_snippets):
        label_submissions[str(pca_di)] = snippet 

    ####### Clustering using Kmeans ###########
    # # HyperTuning Params for Kmeans
    # optimum_num_clusters = kmean_hyper_param_tuning(df)

    # # Fitting KMeans
    # kmeans = KMeans(n_clusters=optimum_num_clusters)
    # kmeans.fit(df)
    # centroids = kmeans.cluster_centers_
    # centroids_pca = pca_transform.transform(centroids)
    
    # # Clustering files (If we just want to see the plot, we can comment this)
    # cluster_files(label_submissions, kmeans.labels_, pca)

    # # Visualize data
    # visualizing_results_3d(pca, kmeans.labels_, centroids_pca)
    ############################################

    ####### Clustering using DBSCAN ###########
    dbscan_model = DBSCAN(eps=0.3, min_samples=5)

    dbscan = dbscan_model.fit(df)
    cluster_files(label_submissions, dbscan.labels_, pca)
    visualizing_results_3d(pca, dbscan.labels_, problem_name, None)
    ############################################

if __name__ == "__main__":
    # TODO: Replace with the data folder (which contains multiple files)
    # Currently, `language` option won't be used.
    # pipeline(folder="/mnt/data/Study/HK232/LVTN/Codes/LNHCodeClustering/data", language="cpp")
    

    pipeline("data", "Stack_1")
