# CuPL
Please run CuPL.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/CuPL.ipynb) or run locally)<br/><br/>
To observe the attention map, please run attn_map.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/attn_map.ipynb) or run locally). The image number is the index number in the dataset test split, whcih is the number printed in the section "Difference In Predictions Between Methods" ("Img#") of CuPL.ipynb.

# CoOp
Please run CoOP.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/CoOP/CoOP.ipynb))<br/><br/>
The configuration provided below was used to generate the results demonstrated in the notebook. To use a different dataset and config, simply modify the parameters accordingly in the CoOP.ipynb

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

# CoCoOp

Please run CoCoOp_Analysis.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/CoCoOp_Analysis.ipynb)). This will train CoCoOp prompt learner on Oxford Flowers downloaded from a Google Drive link, 16 shot for 10 epochs, then generate visualisations afterwards using data saved in `cocoop_prompts.pt`. Runs fastest on A100 (~30 mins). Up to ~150 mins on Colab free.

# Tip_Adapter:

Download the zip file from the mentioned link and extract it. Place the Data folder in this Folder. And run the tip_adapter.ipynb file.

Link: https://drive.google.com/file/d/12timK5svz_vyRgb4YYjzlRBJtmIMI8D8/view?usp=drive_link
