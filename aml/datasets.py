import os
import torchvision
import gdown
import json
import PIL


def download_ucf101(root, download):
    torchvision.datasets.utils.download_and_extract_archive('https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O', os.path.join(root, 'ucf101'), filename='UCF-101-midframes.zip')

# Split datasets (train, val, test) according to https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md
# folder, img folder, json file in Google drive
datasets_list = {
    'caltech101': (torchvision.datasets.Caltech101,     'caltech101',       '101_ObjectCategories', '1hyarUivQE36mY6jSomru6Fjd-JzwcCzN'),
    'oxfordpets': (torchvision.datasets.OxfordIIITPet,  'oxford-iiit-pet',  'images',               '1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs'),
    'flowers102': (torchvision.datasets.Flowers102,     'flowers-102',      'jpg',                  '1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT'),
    'food101'   : (torchvision.datasets.Food101,        'food-101',         'images',               '1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl'),
    'dtd'       : (torchvision.datasets.DTD,            'dtd', os.path.join('dtd', 'images'),       '1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x'),
    'eurosat'   : (torchvision.datasets.EuroSAT,        'eurosat',          '2750',                 '1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o'),
    'ucf101'    : (download_ucf101,                     'ucf101',           'UCF-101-midframes',    '1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y'),
}

class AMLDataset(torchvision.datasets.VisionDataset):
    def __init__(self, dataset_name, root, split: str='train', transforms=None, transform=None, target_transform=None):
        dataset_info = datasets_list[dataset_name]
        dataset_info[0](root, download=True)
        root = os.path.join(root, dataset_info[1])
        super().__init__(root, transforms, transform, target_transform)

        self.img_folder = os.path.join(root, dataset_info[2])
        split_file_path = os.path.join(root, 'split.json')

        if not os.path.exists(split_file_path):
            gdown.download(id=dataset_info[3], output=split_file_path)

        with open(split_file_path, 'r') as f:
            self._items = json.load(f)[split]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        item = self._items[index]

        img = PIL.Image.open(os.path.join(self.img_folder, item[0]))
        target = item[1]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, item[2]

class Caltech101(AMLDataset):
    def __init__(self, *args):
        super().__init__('caltech101', *args)

class OxfordIIITPet(AMLDataset):
    def __init__(self, *args):
        super().__init__('oxfordpets', *args)

class Flowers102(AMLDataset):
    def __init__(self, *args):
        super().__init__('flowers102', *args)

class Food101(AMLDataset):
    def __init__(self, *args):
        super().__init__('food101', *args)

class DTD(AMLDataset):
    def __init__(self, *args):
        super().__init__('dtd', *args)

class EuroSAT(AMLDataset):
    def __init__(self, *args):
        super().__init__('eurosat', *args)

class UCF101(AMLDataset):
    def __init__(self, *args):
        super().__init__('ucf101', *args)
