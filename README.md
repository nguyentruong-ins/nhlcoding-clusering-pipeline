# Set up virtual environment for python
```cmd
python3 -m venv .venv

source ./.venv/bin/activate
```

# Install python packages
```python
pip install -r requirements.txt
```

# Clustering code 
```python
python3 code_cluster.py
```

# Add trained models to pretrained folder
- Download the version that you like in [this link](https://www.kaggle.com/models/trngars/encoder-code-t5p-train-1-epoch/frameworks/PyTorch/variations/train-epoch-1/versions/1)
- Add the file to folder ./pretrained (create one if not exist)

# File usage
- Utilities:
    - `cleaner.py`: Preprocess data
    - `embedding_module.py`: Embedding module which will create embedding vectors for the programs
    - `clustering_module.py`: Clustering module which will use DBSCAN clustering algorithm to cluster the vectors from the embedding_module.py
    - `estimate_dbscan_params.py`: Show plots for combination of some `eps` and `min_samples` values 
- Evaluation:
    - `eval_base_with_finetunes.py`: Eval the base model with finetuned models in order to pick the best model
    - `eval_code.py`: Evaluate and calculate ARI for one problem using single model.
    - `eval_two_model.py`: Evaluate the base model with finetuned model
