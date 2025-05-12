# CoOp
This document helps you run the CoOP.ipynb file located in the CoOp folder.”

Before running the notebook, please update the configuration settings in the first cell.
This notebook demonstrates the results of CoOp on the Caltech101 dataset using the provided configuration.

## Configs

CoOp Configuration
```python
test_dataset_name = 'caltech'   #['airplane', 'caltech', 'dtd', 'flower', 'food', 'pets', 'ucf']
model_name = "ViT-B/16" #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
#CoOp Parameters
NUM_SHOTS = 16
SEED = 1
n_ctx = 16  # few shot learning (1,2,....16)
ctx_init = ""  # context vector, currently not initialized
class_token_position = "end"  #["front", "middle", "end"]
csc = False  # For using the class specific context
input_size = 224  # Input Image Size
```
Training Loop Configuration
```python
MAX_EPOCH = 200
LR = 0.002
MOMENTUM = 0.9
OPTIMIZER = "sgd"
SCHEDULER = "cosine"
```
