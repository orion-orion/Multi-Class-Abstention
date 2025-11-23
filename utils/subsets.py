from torch.utils.data import Dataset


class CustomSubset(Dataset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        # super().__init__(dataset, indices)
        super().__init__()
        self.dataset, self.indices = dataset, indices
        self.subset_transform = subset_transform

    def __getitem__(self, idx):

        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y

    def __len__(self):
        return len(self.indices)
