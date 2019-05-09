import torchvision.datasets as datasets

from torch.utils.data import DataLoader, sampler
from torch.utils.data.sampler import BatchSampler

import os

import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MaxBatchSampler(BatchSampler):
    def __init__(self, targets, batch_size=16, max_num_pos=5):
        self.n_image = len(targets)
        self.target_class = set(targets)  # 505개 0~585
        self.batch_size = batch_size
        self.max_num_pos = max_num_pos

        t = np.array(targets)  # 이미지의 target
        self.idx_per_class = [np.where(t == c)[0] for c in range(max(self.target_class) + 1)]  # class 0~585 의 이미지 인덱스
        self.n_per_class = [len(c) for c in self.idx_per_class]  # class 당 이미지의 수

        self.weak_class=[]      # can't use it as anchor class
        for n,i in enumerate(self.n_per_class):
            if i < 2:
                self.weak_class.append(n)

        self.anchor_class = np.array(list((self.target_class - set(self.weak_class))))


    def __iter__(self):
        out = 0
        pos_classes = np.array(self.anchor_class)
        weak_classes = np.array(self.weak_class).tolist()
        valid_idx_per_class = np.array(self.idx_per_class).tolist()
        n_valid_idx_per_class = np.array(self.n_per_class).tolist()

        while len(pos_classes):
            batch = []
            pos_class = np.random.choice(pos_classes, 1)[0]
            nPos = min(len(valid_idx_per_class[pos_class]), self.max_num_pos)
            nNeg = self.batch_size - nPos
            # print(chose_class, self.idx_per_class[chose_class], nPos, nNeg)

            # get anchor and positive image and remove the entry

            pos_idx = np.random.choice(valid_idx_per_class[pos_class], nPos, replace=False)
            valid_idx_per_class[pos_class] = np.setdiff1d(valid_idx_per_class[pos_class], np.array(pos_idx))
            n_valid_idx_per_class[pos_class] = len(valid_idx_per_class[pos_class])

            batch.extend(pos_idx)
            # consume weak class image first
            weak_pool = []
            for wc in weak_classes:
                if len(valid_idx_per_class[wc]):
                    weak_pool.extend(valid_idx_per_class[wc].tolist())

            if n_valid_idx_per_class[pos_class] <2:
                pos_classes = np.setdiff1d(pos_classes, np.array(pos_class))
                weak_classes.append(pos_class)


            neg_pool = []
            for nc in pos_classes:
                if len(valid_idx_per_class[nc]) and nc != pos_class:
                    neg_pool.extend(valid_idx_per_class[nc].tolist())

            weak_pool = np.array(weak_pool)
            neg_pool = np.array(neg_pool)

            if len(weak_pool) >= nNeg:
                weak_idx = np.random.choice(weak_pool, nNeg, replace=False)
                batch.extend(weak_idx)
            else:
                if len(weak_pool):
                    weak_idx = np.random.choice(weak_pool, len(weak_pool), replace=False)
                    batch.extend(weak_idx)
                neg_cnt = nNeg - len(weak_pool)
                if len(neg_pool) < neg_cnt:
                    neg_cnt = len(neg_pool)

                # for safe
                if len(neg_pool)==0:
                    print('len neg-pool 0 break')
                    break
                neg_idx = np.random.choice(neg_pool, neg_cnt, replace=False)

                batch.extend(neg_idx)

            for n, idxs in enumerate(valid_idx_per_class):
                inc =np.setdiff1d(np.array(idxs), np.array(batch))
                if len(inc) != n_valid_idx_per_class[n]:
                    valid_idx_per_class[n]=np.array(inc)
                    n_valid_idx_per_class[n]=len(inc)
                    if n_valid_idx_per_class[n]<2:
                        pos_classes=np.setdiff1d(pos_classes,np.array(n))
                        weak_classes.append(n)

            out += len(batch)
            yield batch

    def __len__(self):
        return self.n_image // self.batch_size


class ClassBalanceSampler(BatchSampler):
    def __init__(self, targets, batch_size, max_per_class):
        self.targets = targets
        self.n_image = len(targets)
        self.target_class = set(targets)  # 505개 0~585
        self.batch_size = batch_size
        self.max_per_class = max_per_class

    def __iter__(self):
        unused_image = np.arange(self.n_image)
        unused_image_targets = np.array(self.targets)  # 사용되지 않은 이미지의 클래스
        valid_cls = np.array(list(self.target_class))  # 사용 가능한 클래스

        # while len(unused_image_targets):
        while len(unused_image_targets) >= self.batch_size:
            batch = []
            while len(batch) < self.batch_size and len(valid_cls):
                extract_cnt = self.batch_size - len(batch)
                cls = np.random.choice(valid_cls, 1)  # 클래스 하나 뽑아서
                image_idx = np.where(unused_image_targets == cls)[0]  # 이미지의 인덱스 찾음
                available_cnt = min(len(image_idx), self.max_per_class)  # 몇개나 추출가능한지
                if extract_cnt < available_cnt:  # 뽑아야하는 이미지 수가 추출 가능한 이미지 수보다 작다면 변경
                    available_cnt = extract_cnt
                use_idx = np.random.choice(image_idx, available_cnt, replace=False)  # 추출예정 - 배치사이즈 초과시
                batch.extend(unused_image[use_idx])
                unused_image = np.delete(unused_image, use_idx)
                unused_image_targets = np.delete(unused_image_targets, use_idx)
                valid_cls = np.array(list(set(unused_image_targets)))
            if len(set(np.array(self.targets)[batch])) < 2:
                continue
            yield batch

    def __len__(self):
        return self.n_image // self.batch_size

if __name__ == '__main__':

    class myFolder(datasets.DatasetFolder):
        def __init__(self, root, loader, extensions, transform=None, target_transform=None):
            super(myFolder, self).__init__(root, loader, extensions, transform, target_transform)

        def __getitem__(self, index):
            path, target = self.samples[index]

            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target, path

    import torchvision.transforms as trn

    import models.nets
    import models.pooling
    import torch.autograd.variable as V
    from losses import OnlineTripletLoss, HardestNegativeTripletSelector

    test_trn = trn.Compose([
        trn.Resize((32, 32)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dt = myFolder(os.path.join('/data', 'landmark', 'Landmark-clean', 'valid'),
                  datasets.folder.default_loader,
                  ['jpg'], transform=test_trn)

    from . import OnlineTripletLoss, HardestNegativeTripletSelector

    criterion = OnlineTripletLoss(0.8, HardestNegativeTripletSelector(0.8, True))
    net = models.nets.Basic(models.pooling.RMAC())
    dl = DataLoader(dt, batch_sampler=MaxBatchSampler(dt.targets), num_workers=8)

    for i in range(10):
        for n, i in enumerate(dl):
            print(n, i[2])
            # a, m = net(V(i[0]))
            # l, dp,dn,t = criterion(a, i[1])
