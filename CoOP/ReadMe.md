# CoOp
This document helps you run the CoOP.ipynb file.”

Before running the notebook, please update the configuration settings in the first cell.
This notebook demonstrates the results of CoOp on the Caltech101 dataset using the provided configuration.

## Configs

CoOp Configuration
```python
test_dataset_name = 'caltech' 
model_name = "ViT-B/16"
NUM_SHOTS = 16
SEED = 1
n_ctx = 16 
ctx_init = "" 
class_token_position = "end"  
csc = False 
input_size = 224 
```
Training Loop Configuration
```python
MAX_EPOCH = 200
LR = 0.002
MOMENTUM = 0.9
OPTIMIZER = "sgd"
SCHEDULER = "cosine"
```
