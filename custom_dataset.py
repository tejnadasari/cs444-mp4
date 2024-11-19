from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels= None, transforms = None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[index])
        else:
            return data