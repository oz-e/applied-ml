# CuPL
Please run CuPL.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/CuPL.ipynb))<br/><br/>
To observe the attention map, please run attn_map.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/attn_map.ipynb)). The image number is the index number in the dataset test split, whcih is the number printed in the section "Difference In Predictions Between Methods" ("Img#") of CuPL.ipynb.

# CoOp
Please run CoOP.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/CoOP/CoOP.ipynb))<br/><br/>
View CoOP/ReadMe.md for CoOp implmentation details and configuations.

# CoCoOp

Please run CoCoOp_Analysis.ipynb ([open in Colab](https://colab.research.google.com/github/oz-e/applied-ml/blob/main/CoCoOp_Analysis.ipynb)). This will train CoCoOp prompt learner on Oxford Flowers downloaded from a Google Drive link, 16 shot for 10 epochs, then generate visualisations afterwards using data saved in `cocoop_prompts.pt`. Runs fastest on A100 (~30 mins). Up to ~150 mins on Colab free.
