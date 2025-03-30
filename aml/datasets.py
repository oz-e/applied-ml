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

        # Download the dataset with the help of torchvision.datasets object
        dataset_info[0](root, download=True)

        # Since torchvision.datasets put data inside a subfolder, we change path into this new root folder, and store everything inside it
        root = os.path.join(root, dataset_info[1])
        super().__init__(root, transforms, transform, target_transform)

        # Images are further inside the new root folder
        self.img_folder = os.path.join(root, dataset_info[2])

        # Download json inside the new root folder
        split_file_path = os.path.join(root, 'split.json')
        if not os.path.exists(split_file_path):
            gdown.download(id=dataset_info[3], output=split_file_path)

        # Read json file, resulting in a dict[str('train', 'val', 'test'), list[str(impath), int(label), str(classname)]]
        with open(split_file_path, 'r') as f:
            data_source = json.load(f)

        self._items = data_source[split]
        self._num_classes = self.get_num_classes(data_source['train'])
        self._lab2cname, self._classnames = self.get_lab2cname(data_source['train'])

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        impath, label, classname = self._items[index]

        img = PIL.Image.open(os.path.join(self.img_folder, impath))

        # if self.transforms is not None:
        #     img, label = self.transforms(img, label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for impath, label, classname in data_source:
            label_set.add(label)
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for impath, label, classname in data_source:
            container.add((label, classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames


class Caltech101(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('caltech101', *args, **kwargs)


class OxfordIIITPet(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('oxfordpets', *args, **kwargs)


class Flowers102(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('flowers102', *args, **kwargs)


class Food101(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('food101', *args, **kwargs)


class DTD(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('dtd', *args, **kwargs)


class EuroSAT(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('eurosat', *args, **kwargs)


class UCF101(AMLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__('ucf101', *args, **kwargs)


class FGVCAircraft(torchvision.datasets.FGVCAircraft):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, download=True, **kwargs)

    @property
    def lab2cname(self):
        return self.classes

    @property
    def classnames(self):
        return self.classes

    @property
    def num_classes(self):
        return len(self.classes)
