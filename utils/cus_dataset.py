from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    '''A custom dataset class with customizable data transformation'''

    def __init__(self, X, y, subset_transform=None):
        # super().__init__(dataset, indices)
        super().__init__()
        self.X, self.y = X, y
        self.subset_transform = subset_transform

    def __getitem__(self, idx):

        X, y = self.X[idx], self.y[idx]

        if self.subset_transform:
            X = self.subset_transform(X)

        return X, y

    def __len__(self):
        return len(self.y)


class CustomSubset(Dataset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        # super().__init__(dataset, indices)
        super().__init__()
        self.dataset, self.indices = dataset, indices
        self.subset_transform = subset_transform

    def __getitem__(self, idx):

        X, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            X = self.subset_transform(X)

        return X, y

    def __len__(self):
        return len(self.indices)
