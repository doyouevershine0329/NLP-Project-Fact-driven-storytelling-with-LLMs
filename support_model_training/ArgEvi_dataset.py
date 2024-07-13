import pandas as pd
from torch.utils.data import Dataset


class ArgEvi(Dataset):

    def __init__(self, file_path):
        data = pd.read_csv(file_path)

        data = data.values.tolist()

        print(data[0])

        self.data = data
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

#file_path = 'pos_neg_pairs_train.csv'
#dataset = ArgEvi(file_path)
