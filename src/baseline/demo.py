import pickle
import pdb
import torch 
import numpy as np 
from torch.utils.data import DataLoader, random_split

class Data:

    def __init__(self, args, data_paths, mode = 'train'):

        #read the files
        self.args = args
        self.data = []
        self.labels = []
        self.device = self.args.device

        for file_path in data_paths:
            
            #dictionary containgin
            #b'data', b'labels'
            batch_data = self.unpickle(file_path)
            batch_data[b'data'] = torch.tensor(batch_data[b'data'].reshape(-1,3,32,32)).float()
            self.data.append(batch_data[b'data'])
            self.labels.append(batch_data[b'labels'])
        pdb.set_trace()
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.tensor([0.247, 0.243, 0.261]).unsqueeze(-1).unsqueeze(-1).cuda()

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __getitem__(self, index):
        batch = index // 10000
        index = index % 10000

        data = self.data[batch][index].to(self.device)
        
        data = (data/255 - self.mean)/self.std
        label = torch.tensor(self.labels[batch][index]).to(self.device)
        return data, label

    def __len__(self):
        return sum([len(d) for d in self.data])

'''
returns dataloader depending on mode
if mode == train:
    if val_split is not none:
        train, val
    else:
        train
else:
    test
'''
# #the collat function
# def collate_fn_help(device, batch):
#     pdb.set_trace()
#     data = batch[0].to(device)
#     labels = batch[1].to(device)
#     return data, labels

def get_dataloader(args, mode):

    #get data
    if(mode == 'train'):
        data = Data(args, args.data)
    else:
        data = Data(args, args.test_data)


    #if train then check for splitting
    if(mode == 'train' and args.val_split is not None):
        
        #split the data according to seed
        seed = torch.Generator().manual_seed(42)
        val_len = int(args.val_split*len(data))
        datas = random_split(data, [val_len, len(data) - val_len], generator=seed)

        label_dict = dict()
        for i in range(10):
            label_dict[i] = 0
        for i in datas[0].indices:
            label_dict[datas[0].dataset[i][1].item()] += 1

        #return the dataloaders
        val_data = DataLoader(datas[0], batch_size = args.train.batch_size, shuffle = True)#, collate_fn = collate_fn)
        train_data = DataLoader(datas[1], batch_size = args.train.batch_size, shuffle = True)#, collate_fn = collate_fn)
        return val_data, train_data

    #else just return the normal data
    data = DataLoader(data, batch_size = args[mode].batch_size, shuffle = True)#, collate_fn = collate_fn)
    return data

from config import get_config
args = get_config([])
t = Data(args,args.data, )