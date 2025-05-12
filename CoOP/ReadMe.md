# CoOp
This document helps u run the CoOP.ipynb file under "CoOP/"

Before running the notebook, set the model configurations in the beginning of the notebook. (The notebook displays the result of CoOP on caltech with exisiting Configuration.)

## Configs

CoOp Configuration
```python
test_dataset_name = 'caltech'   #['airplane', 'caltech', 'dtd', 'flower', 'food', 'pets', 'ucf']
model_name = "ViT-B/16" #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
# mention all the parameters
NUM_SHOTS = 16
SEED = 1
n_ctx = 16  # few shot learning (1,2,....16)
ctx_init = ""  # context vector, rn its not initialized
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
