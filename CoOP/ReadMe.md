# CoOp

The configuration provided below was used to generate the results demonstrated in this notebook. To use a different dataset, simply modify the parameters accordingly.

## Configs

CoOp Configuration
```python
test_dataset_name = 'caltech'       # Dataset name
model_name = 'ViT-B/16'             # Vision Transformer model
NUM_SHOTS = 16                      # Number of shots for few-shot learning
SEED = 1                            # Random seed for reproducibility
n_ctx = 16                          # Number of context tokens
ctx_init = ""                        # Initial context string
class_token_position = "end"        # Position of the class token
csc = False                         # Use class-specific context
input_size = 224                    # Image input size
```
Training Loop Configuration
```python
MAX_EPOCH = 200                    # Number of training epochs
LR = 0.002                          # Learning rate
MOMENTUM = 0.9                      # Momentum for the optimizer
OPTIMIZER = 'sgd'                   # Optimizer type
SCHEDULER = 'cosine'                # Learning rate scheduler
```
