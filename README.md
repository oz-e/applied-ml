# About Using Datasets
Code is created to create dataset object just like `torchvision.datasets`. It is split according to the json of CoOp (https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md), ensuring consistent results:
```python3
import aml.datasets


caltech101_dataset_train = aml.datasets.Caltech101(datasets_path, split='train')
oxfordpets_dataset_train = aml.datasets.OxfordIIITPet(datasets_path, split='train')
flowers102_dataset_train = aml.datasets.Flowers102(datasets_path, split='train')
food101_dataset = aml.datasets.Food101(datasets_path, split='train')
dtd_dataset_train = aml.datasets.DTD(datasets_path, split='train')
eurosat_dataset_train = aml.datasets.EuroSAT(datasets_path, split='train')
ucf101_dataset_train = aml.datasets.UCF101(datasets_path, split='train')
```

To get class names from label, use the dict `lab2cname`\
To get a sorted list of class names, use the list `classnames`\
To get total num of classes, use `num_classes`


# CuPL
caltech:\
Top-1 accuracy standard: 93.06\
Top-1 accuracy CuPL: 94.20\
Top-1 accuracy both: 93.67

pets:\
Top-1 accuracy standard: 87.05\
Top-1 accuracy CuPL: 91.22\
Top-1 accuracy both: 88.53

flower:\
Top-1 accuracy standard: 65.98\
Top-1 accuracy CuPL: 73.93\
Top-1 accuracy both: 68.98

food:\
Top-1 accuracy standard: 83.84\
Top-1 accuracy CuPL: 86.02\
Top-1 accuracy both: 85.06

airplane:\
Top-1 accuracy standard: 23.64\
Top-1 accuracy CuPL: 27.93\
Top-1 accuracy both: 25.68

dtd:\
Top-1 accuracy standard: 45.04\
Top-1 accuracy CuPL: 53.43\
Top-1 accuracy both: 48.17

ucf:\
Top-1 accuracy standard: 65.08\
Top-1 accuracy CuPL: 70.34\
Top-1 accuracy both: 68.09
