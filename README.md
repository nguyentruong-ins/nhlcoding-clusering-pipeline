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
