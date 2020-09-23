from torch.utils.data import Dataset
import numpy as np 
import torch

from extract_features import make_grid, rotate

class DB(Dataset):
    """Accent dataset."""

    def __init__(self, coords, features, affinity, dataset_name, rot, std):
        self.coords = coords[dataset_name]
        self.raw_features = features[dataset_name]

        self.rot = rot
        self.std = std

        self.affinity = affinity[dataset_name]
        self.affinity = np.stack(self.affinity)
        self.affinity = torch.FloatTensor(np.float32(self.affinity))

        self.features_pocket = list()
        self.features_ligand = list()
        self.__transform()

        del self.raw_features, self.rot, self.std, self.coords

        
        self.features_pocket = torch.from_numpy(self.features_pocket)
        self.features_pocket = torch.squeeze(self.features_pocket, 1).permute(0, 4, 1, 2, 3)
        self.features_ligand = torch.from_numpy(self.features_ligand)
        
        
    def __transform(self):
        for idx in range(self.affinity.shape[0]):
            coords_idx = rotate(self.coords[idx], self.rot)
            features_idx = self.raw_features[idx]['pocket']
            self.features_pocket.append(make_grid(coords_idx, features_idx, grid_resolution=1, max_dist=10))
            self.features_ligand.append(self.raw_features[idx]['ligand'])
        self.features_pocket = np.stack(self.features_pocket)
        self.features_ligand = np.stack(self.features_ligand)
        self.features_pocket[..., 12] /= self.std

    
    def __len__(self):
        return int(self.affinity.shape[0])

    def __getitem__(self, idx):
        return self.features_pocket[idx].type(torch.FloatTensor), self.features_ligand[idx].type(torch.FloatTensor), self.affinity[idx]

import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, num_classes=1):
        super(net, self).__init__()
        self.features_pocket = nn.Sequential(
            nn.Conv3d(43, 64, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, padding=(1,1,1)),
            nn.Conv3d(64, 128, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, 5, padding=(1,1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(256),
            # nn.Conv3d(256, 512, 3, padding=(1,1,1)),
            # nn.ReLU(inplace = True),
            # nn.MaxPool3d(2),
            # nn.BatchNorm3d(512),
        )

        self.features_ligand = nn.Sequential(
            nn.Linear(11496, 7000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(7000, 5000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(5000, 2000),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256*(3**3)+ 2000, 7000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(15000, 7000),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(7000, 2000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 1)
        )

    def forward(self, x_p, x_l):
        x_p = self.features_pocket(x_p)
        x_l = self.features_ligand(x_l)
        # print(x_p.shape, x_l.shape)
        
        x_p = x_p.view(x_p.size()[0], 256*(3**3))
        x = torch.cat((x_p, x_l),1)
        # print(x.shape)
        del x_p, x_l
        x = self.regressor(x)
        return x

class net1(nn.Module):
    def __init__(self, num_classes=1):
        super(net1, self).__init__()
        self.features_pocket = nn.Sequential(
            nn.Conv3d(43, 64, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, padding=(1,1,1)),
            nn.Conv3d(64, 128, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, 3, padding=(1,1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 512, 3, padding=(1,1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(512),
        )

        self.features_pocket_attn = nn.Sequential(
            nn.Linear(512*(2**3), 512*(2**3)),
            nn.ReLU(inplace = True),
        )

        self.features_ligand = nn.Sequential(
            nn.Linear(11496, 7000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(7000, 5000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(5000, 5000),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512*(2**3)+ 5000, 7000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(15000, 7000),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(7000, 3000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(3000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 1)
        )

    def forward(self, x_p, x_l):
        x_p = self.features_pocket(x_p)
        x_l = self.features_ligand(x_l)
        # print(x_p.shape, x_l.shape)
        
        x_p = x_p.view(x_p.size()[0], 512*(2**3))
        x_p = self.features_pocket_attn(x_p)
        x = torch.cat((x_p, x_l),1)
        # print(x.shape)
        del x_p, x_l
        x = self.regressor(x)
        return x


class net2(nn.Module):
    def __init__(self, num_classes=1):
        super(net1, self).__init__()
        self.features_pocket = nn.Sequential(
            nn.Conv3d(43, 64, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, padding=(1,1,1)),
            nn.Conv3d(64, 128, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, 3, padding=(1,1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 512, 3, padding=(1,1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(512),
        )

        self.features_pocket_attn = nn.Sequential(
            nn.Linear(512*(2**3), 512*(2**3)),
            nn.Softmax(),
        )

        self.features_ligand = nn.Sequential(
            nn.Linear(11496, 7000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(7000, 5000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(5000, 5000),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512*(2**3)+ 5000, 7000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(15000, 7000),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(7000, 3000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(3000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 1)
        )

    def forward(self, x_p, x_l):
        x_p = self.features_pocket(x_p)
        x_l = self.features_ligand(x_l)
        # print(x_p.shape, x_l.shape)
        
        x_p = x_p.view(x_p.size()[0], 512*(2**3))
        x_p = torch.mul(self.features_pocket_attn(x_p), x_p)
        x = torch.cat((x_p, x_l),1)
        # print(x.shape)
        del x_p, x_l
        x = self.regressor(x)
        return x


class trail_net(nn.Module):
    def __init__(self,in_channels = 19, conv_filters = [32, 64, 128, 256], conv_patch = 5, dense_layers = [1000, 500, 250], pool_patch = 2):
        super(trail_net, self).__init__()
        conv_modules = []
        for idx, c_filter in enumerate(conv_filters):
            if(idx == 0):
                conv_modules.append(nn.Conv3d(in_channels, c_filter, kernel_size = (conv_patch, conv_patch, conv_patch), padding = (3, 3, 3), padding_mode='zeros'))
                conv_modules.append(nn.ReLU(inplace = True))
                conv_modules.append(nn.MaxPool3d(kernel_size = (pool_patch, pool_patch, pool_patch), padding = (1, 1, 1)))
            else:
                conv_modules.append(nn.Conv3d(conv_filters[idx-1], c_filter, kernel_size = (conv_patch, conv_patch, conv_patch), padding = (3, 3, 3), padding_mode='zeros'))
                conv_modules.append(nn.ReLU(inplace = True))
                conv_modules.append(nn.MaxPool3d(kernel_size = (pool_patch, pool_patch, pool_patch), padding = (1, 1, 1)))
                conv_modules.append(nn.BatchNorm3d(c_filter))

        self.features = nn.Sequential(*conv_modules)
        del conv_modules

        self.flatten_size = conv_filters[-1] * (self.__calc_outsize(21, conv_filters, conv_patch, pool_patch) ** 3)
        dense_modules = []
        for idx, d_layer in enumerate(dense_layers):
            if(idx == 0):
                dense_modules.append(nn.Linear(self.flatten_size, d_layer))
                dense_modules.append(nn.ReLU(inplace = True))
                dense_modules.append(nn.Dropout(0.5))
            else:
                dense_modules.append(nn.Linear(dense_layers[idx-1], d_layer))
                dense_modules.append(nn.ReLU(inplace = True))
                dense_modules.append(nn.Dropout(0.5))

        dense_modules.append(nn.Linear(dense_layers[-1], 1))

        self.regressor = nn.Sequential(*dense_modules)
        del dense_modules
    def __calc_outsize(self, in_size, conv_filters, conv_patch, pool_patch):
        stride_c, stride_p = 1, 1
        padding_c, padding_p = 0,0
        dilation_c, dilation_p = 1, 1
        out_size = in_size
        for idx, c_filter in enumerate(conv_filters):
            out_size = ((in_size + (2 * padding_c) - dilation_c * (conv_patch - 1) - 1))/stride_c + 1
            # out_size = (((in_size - conv_patch + (2 * padding_c)))/stride) + 1
            out_size_m = ((out_size + (2 * padding_p) - dilation_p * (pool_patch - 1) - 1))/stride_p + 1
            # out_size_m = (((out_size - pool_patch + (2 * padding_p)))/stride) + 1
            print(out_size, out_size_m)
            in_size = out_size_m
        return int(out_size_m)

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size()[0], self.flatten_size)
        x = self.regressor(x)
        return x

class netpdb(nn.Module):
    def __init__(self, num_classes=1):
        super(netpdb, self).__init__()
        self.features_pocket = nn.Sequential(
            nn.Conv3d(43, 64, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            # nn.MaxPool3d(2, padding=(1,1,1)),
            nn.Conv3d(64, 128, 5, padding=(3,3,3)),
            nn.ReLU(inplace = True),
            # nn.MaxPool3d(2, padding=(1,1,1)),
            nn.BatchNorm3d(128),
            # nn.Conv3d(128, 256, 5, padding=(1,1,1)),
            # nn.ReLU(inplace = True),
            # nn.MaxPool3d(2),
            # nn.BatchNorm3d(256),
            # nn.Conv3d(256, 512, 3, padding=(1,1,1)),
            # nn.ReLU(inplace = True),
            # nn.MaxPool3d(2),
            # nn.BatchNorm3d(512),
        )

        self.features_ligand = nn.Sequential(
            nn.Linear(11496, 1000),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(5000, 2000),
            # nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256*(3**3)+ 200, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # nn.Linear(15000, 7000),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(7000, 2000),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(7000, 500),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(500, 200),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(500, 1)
        )

    def forward(self, x_p, x_l):
        x_p = self.features_pocket(x_p)
        x_l = self.features_ligand(x_l)
        print(x_p.shape, x_l.shape)
        
        x_p = x_p.view(x_p.size()[0], 256*(3**3))
        x = torch.cat((x_p, x_l),1)
        # print(x.shape)
        del x_p, x_l
        x = self.regressor(x)
        return x