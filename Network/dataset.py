from torchvision.datasets.folder import default_loader
from torchvision.datasets import DatasetFolder
import torchvision.transforms as trn
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset

from random import shuffle
import numpy as np
from PIL import Image


class TripletDataset(DatasetFolder):
    def __init__(self, root, transform=None):
        self.transform = trn.ToTensor() if transform == None else transform
        super(TripletDataset, self).__init__(root, default_loader, ['.jpg'], transform=self.transform,
                                             target_transform=None)

    def __getitem__(self, index):
        path, target = self.samples[index]

        pos = list(filter(lambda x: x[1] == target, self.samples))
        shuffle(pos)
        neg = list(filter(lambda x: x[1] != target, self.samples))
        shuffle(neg)

        p_path, p_target = pos[0]
        n_path, n_target = neg[0]

        try:
            sample = self.loader(path)
            p_sample = self.loader(p_path)
            n_sample = self.loader(n_path)

            if self.transform is not None:
                sample = self.transform(sample)
                p_sample = self.transform(p_sample)
                n_sample = self.transform(n_sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
                p_target = self.target_transform(p_target)
                n_target = self.target_transform(n_target)

            return (sample, target, path), (p_sample, p_target, p_path), (n_sample, n_target, n_path)
        except Exception as e:
            # print('next idx ', path, p_path, n_path, e, index, file=sys.stderr)
            return self.__getitem__(index + 1) if index + 1 != len(self.samples) else self.__getitem__(0)


class ListDataSet(Dataset):
    def __init__(self, imlist, image_size=None, bbx=None, transform=None):
        super(ListDataSet, self).__init__()

        self.transform = trn.ToTensor() if transform == None else transform
        self.imlist = imlist
        self.loader = default_loader
        self.bbx = bbx
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imlist[index]
        sample = self.loader(path)

        if self.bbx is not None:
            bbx = self.bbx[index]
            sample = sample.crop(bbx)
        if self.image_size is not None:
            sample.thumbnail((self.image_size, self.image_size), Image.ANTIALIAS)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.imlist)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class defaultDataset(DatasetFolder):
    def __init__(self, root, transform=None):
        self.transform = trn.ToTensor() if transform == None else transform
        super(defaultDataset, self).__init__(root, default_loader, ['.jpg'], transform=self.transform,
                                             target_transform=None)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target, path

        except Exception as e:
            return self.__getitem__(index + 1) if index + 1 != len(self.samples) else self.__getitem__(0)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader

    transform_valid = transform_train = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainSet = defaultDataset(os.path.join('/data', 'landmark', 'Landmark-clean', 'valid'),
                              transform=transform_train)

    print(trainSet)
    trainLoader = DataLoader(trainSet, batch_size=50, shuffle=False, num_workers=0)

    for i in trainLoader:
        print(i[1], i[2])
