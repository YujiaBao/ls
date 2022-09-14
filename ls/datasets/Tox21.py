import os
from os.path import join
import csv
import gzip

from scipy import io
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from ls.utils.print import print


class Tox21(Dataset):
    def __init__(self,
                 root: str = './datasets/tox21',
                 task: str = 'NR.AR'):
        '''
            We use the Tox21 Machine Learning Data Set published by
            http://bioinf.jku.at/research/DeepTox/tox21.html

            root: path to download/load the data.
            task: a specific property prediction task that we want to load.
                Options: ["NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase",
                "NR.ER", "NR.ER.LBD", "NR.PPAR.gamma", "SR.ARE", "SR.ATAD5",
                "SR.HSE", "SR.MMP", "SR.p53"]
        '''
        if not os.path.isdir(root):
            # The tox21 data dir doesn't exist. Download the dataset.
            Tox21.download(root)

        assert task in ["NR.AhR", "NR.AR", "NR.AR.LBD", "NR.Aromatase",
                        "NR.ER", "NR.ER.LBD", "NR.PPAR.gamma", "SR.ARE",
                        "SR.ATAD5", "SR.HSE", "SR.MMP", "SR.p53" ], \
            f"Task {task} is not supported in Tox21."

        x, y = Tox21.load_data(root, task)

        # Normalize the features.
        x = torch.tensor(StandardScaler().fit_transform(x)).float()

        self.data = x
        self.targets = y
        self.length = len(self.targets)

    @staticmethod
    def download(root: str):
        '''
            Download Tox21 data and save it under the root
        '''
        os.makedirs(root, exist_ok=True)

        # Files the we need.
        url_list = [
            'http://bioinf.jku.at/research/DeepTox/tox21_dense_train.csv.gz',
            'http://bioinf.jku.at/research/DeepTox/tox21_sparse_train.mtx.gz',
            'http://bioinf.jku.at/research/DeepTox/tox21_dense_test.csv.gz',
            'http://bioinf.jku.at/research/DeepTox/tox21_sparse_test.mtx.gz',
            'http://bioinf.jku.at/research/DeepTox/tox21_labels_train.csv.gz',
            'http://bioinf.jku.at/research/DeepTox/tox21_labels_test.csv.gz',
        ]

        # https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
        import shutil
        from urllib.request import urlopen

        print(f"Downloading the Tox21 dataset to {root}")
        for url in url_list:
            save_path = join(root, url.split('/')[-1])

            print(f"Saving {url}")
            with urlopen(url) as response, open(save_path, 'wb') as f:
                shutil.copyfileobj(response, f)

        print("Done")

    @staticmethod
    def load_data(root: str, task: str):
        '''
            Load the Tox21 dataset files.
            The procedures are mostly the same as in
            http://bioinf.jku.at/research/DeepTox/sampleCode.py
            We remove the dependencies on pandas.
        '''
        # We will combine the original train and test splits together.
        #
        # Step 1. Collect the label for the task of interest
        y = []
        for label_file in ['tox21_labels_train.csv.gz',
                           'tox21_labels_test.csv.gz']:

            with gzip.open(join(root, label_file), mode='rt') as f:

                reader = csv.DictReader(f)
                for row in reader:
                    if row[task] == '0':
                        y.append(0)
                    elif row[task] == '1':
                        y.append(1)
                    else:
                        # This example is not annotated for the task of
                        # interest. We will mask out this example later.
                        y.append(-1)

        y = torch.tensor(y).long()

        # Step 2. Collect the dense feature
        x_dense = []
        keys = None
        for dense_file in ['tox21_dense_train.csv.gz',
                           'tox21_dense_test.csv.gz']:
            with gzip.open(join(root, dense_file), mode='rt') as f:

                reader = csv.DictReader(f)
                for row in reader:
                    if keys is None:
                        # features start from the second column in the csv file
                        keys = list(row.keys())[1:]

                    x_dense.append([float(row[k]) for k in keys])
        x_dense = torch.tensor(x_dense)

        # Step 3. Collect the sparse feature
        x_tr_sparse = io.mmread(join(root, 'tox21_sparse_train.mtx.gz')).tocsc()
        x_te_sparse = io.mmread(join(root, 'tox21_sparse_test.mtx.gz')).tocsc()

        sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()

        x_sparse = torch.cat([
            torch.from_numpy(x_tr_sparse[:, sparse_col_idx].A),
            torch.from_numpy(x_te_sparse[:, sparse_col_idx].A)
        ], dim=0)

        # Step 4. Combine the sparse and dense features
        x = torch.cat([x_dense, x_sparse], dim=1).float()

        # Step 5. Mask out examples that are not available for the current
        # task
        x = x[y >= 0]
        y = y[y >= 0]

        return x, y

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Return the molecule representation and the label for the given
            index.
        '''
        return self.data[idx], self.targets[idx]
