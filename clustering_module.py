from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import collections
import os
import shutil
import pandas as pd

class Clusterer():
    def __init__(self) -> None:
        pass

    def pca_embeddings(self, df, dimension=2):
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

    def visualizing_results_3d(self, pca_result, label, problem_name = None, centroids_pca = None):
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

        ax.legend()

        plt.show()

    def cluster_files(self, data_locate, kmeans_lables, pca):
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
    
    def cluster_process(self, df):
        ####### Clustering using DBSCAN ###########
        dbscan_model = DBSCAN(eps=0.25, min_samples=3)

        dbscan = dbscan_model.fit(df)
        ############################################

        return dbscan