# This file is for full pipeline of code embedding and clustering
# Just for evaluate

import pandas as pd
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from transformers import AutoModel
from model import CodeT5ClassificationModel
from embedding_module import Embedder
from clustering_module import Clusterer
from PyQt5.QtWidgets import (
                        QWidget,
                        QApplication,
                        QMainWindow,
                        QVBoxLayout,
                        QScrollArea,
                    )

from matplotlib.backends.backend_qt5agg import (
                        FigureCanvasQTAgg as FigCanvas,
                        NavigationToolbar2QT as NabToolbar,
                    )

## SET UP PYQ5T
class MyApp(QWidget):
    def __init__(self, fig):
        super().__init__()
        self.title = 'VERTICAL, HORIZONTAL SCROLLABLE WINDOW : HERE!'
        self.posXY = (0, 0)
        self.windowSize = (2000, 1200)
        self.fig = fig
        self.initUI()

    def initUI(self):
        QMainWindow().setCentralWidget(QWidget())

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        canvas = FigCanvas(self.fig)
        canvas.draw()

        scroll = QScrollArea(self)
        scroll.setWidget(canvas)

        nav = NabToolbar(canvas, self)
        self.layout().addWidget(nav)
        self.layout().addWidget(scroll)

        self.show_basic()

    def show_basic(self):
        self.setWindowTitle(self.title)
        self.setGeometry(*self.posXY, *self.windowSize)
        self.show()

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
    vectors, program_snippets, _, testcase_results = embedder.load_embeddings_csv(file, slug, get_testcase=True)

    input_vectors = None
    if (len(testcase_results) == 0):
        input_vectors = vectors
    else:
        added_testcase_vectors = [np.concatenate([vector, np.array(testcase_result) / 15]) for vector, testcase_result in zip(vectors, testcase_results)]
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
    dbscan = clusterer.dbscan_cluster_process(df, eps=0.2, min_samples=4)

    # Clustering files
    # clusterer.cluster_files(label_submissions, dbscan.labels_, pca)


    # Test the results
    min_samples_range = range(2, 4)
    eps_range = [ele/100 for ele in range(10, 55, 5)]

    num_cols =  len(min_samples_range)
    num_rows = len(eps_range)

    fig = plt.figure(figsize=(12, 4*num_rows))

    index = 1
    for loop_eps in eps_range:
        for loop_min_samples in min_samples_range:
            dbscan = clusterer.dbscan_cluster_process(df, eps=loop_eps, min_samples=loop_min_samples)

            x = pca[:, 0]
            y = pca[:, 1]
            z = pca[:, 2]

            # fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot(num_rows, num_cols, index, projection = "3d")

            scatter = ax.scatter3D(x, y, z, c = dbscan.labels_)
            # Adding labels for data points
            legend = ax.legend(*scatter.legend_elements(),
                            title="Legend", loc='upper left')
            ax.add_artist(legend)
            
            for i in range(len(x)):
                ax.text(x[i], y[i], z[i], '%s' % (str(i)), size=13, zorder=1)

            ax.legend()
            ax.set_title("min_samples={}, eps={}".format(loop_min_samples, loop_eps))

            index += 1

    app = QApplication(sys.argv)
    window = MyApp(fig)
    sys.exit(app.exec_())

if __name__ == "__main__":
    pipeline("csv/processed_v7_labeled_v3.csv", "Sorting_1")
