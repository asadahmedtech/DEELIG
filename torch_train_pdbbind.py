import numpy as np
np.random.seed(123)

import pandas as pd
from math import sqrt, ceil

import h5py

from sklearn.utils import shuffle

from extract_features import make_grid, rotate
import os.path

from torch_files import *
from torch_utils import * 
# import matplotlib as mpl
# mpl.use('agg')

# import seaborn as sns
# sns.set_style('white')
# sns.set_context('paper')
# sns.set_color_codes()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import gc

import time
timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')


datasets = ['training', 'validation', 'test']
NaN_present = ['1D2V_BR_A_601', '1D2V_BR_A_758', '1D2V_BR_A_843', '1D2V_BR_A_889', '1D2V_BR_B_601', '1D2V_BR_B_758', '1D2V_BR_B_843', 
              '1D2V_BR_B_889', '1IXI_2HP_A_322', '2HAW_2PN_A_2001', '2HAW_2PN_B_2002', '2IW4_2PN_A_1315', '3IAI_PO4_A_600', '3IAI_PO4_A_601', 
              '3IAI_PO4_B_600', '3IAI_PO4_C_600', '3IAI_PO4_D_600', '3QUG_GIX_A_700', '3QUG_GIX_B_700', '3TVL_3PO_B_231', '4H5D_POP_F_402', 
              '2IW4_2PN_B_1318', '4KII_RHL_A_201', '3TVL_3PO_A_231']

import argparse
parser = argparse.ArgumentParser(
    description='Train 3D colnvolutional neural network on affinity data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

io_group = parser.add_argument_group('I/O')
io_group.add_argument('--input_dir', '-i',default = '../dataset',
                      help='directory with training, validation and test sets')
io_group.add_argument('--log_dir', '-l', default='./logdir/',
                      help='directory to store tensorboard summaries')
io_group.add_argument('--output_prefix', '-o', default='./output',
                      help='prefix for checkpoints, predictions and plots')
io_group.add_argument('--grid_spacing', '-g', default=1.0, type=float,
                      help='distance between grid points')
io_group.add_argument('--max_dist', '-d', default=10.0, type=float,
                      help='max distance from complex center')


arc_group = parser.add_argument_group('Netwrok architecture')
arc_group.add_argument('--conv_patch', default=5, type=int,
                       help='patch size for convolutional layers')
arc_group.add_argument('--pool_patch', default=2, type=int,
                       help='patch size for pooling layers')
arc_group.add_argument('--conv_channels', metavar='C', default=[64, 128, 256],
                       type=int, nargs='+',
                       help='number of fileters in convolutional layers')
arc_group.add_argument('--dense_sizes', metavar='D', default=[1000, 500, 200],
                       type=int, nargs='+',
                       help='number of neurons in dense layers')

reg_group = parser.add_argument_group('Regularization')
reg_group.add_argument('--keep_prob', dest='kp', default=0.5, type=float,
                       help='keep probability for dropout')
reg_group.add_argument('--l2', dest='lmbda', default=0.001, type=float,
                       help='lambda for weight decay')
reg_group.add_argument('--rotations', metavar='R', default=list(range(24)),
                       type=int, nargs='+',
                       help='rotations to perform')

tr_group = parser.add_argument_group('Training')
tr_group.add_argument('--learning_rate', default=1e-5, type=float,
                      help='learning rate')
tr_group.add_argument('--batch_size', default=20, type=int,
                      help='batch size')
tr_group.add_argument('--num_epochs', default=20, type=int,
                      help='number of epochs')
tr_group.add_argument('--num_checkpoints', dest='to_keep', default=4, type=int,
                      help='number of checkpoints to keep')
tr_group.add_argument('--resume', default=0, type=bool,
                      help='To reumse the training')

args = parser.parse_args()

prefix = os.path.abspath(args.output_prefix) + '-' + timestamp
logdir = os.path.join(os.path.abspath(args.log_dir), os.path.split(prefix)[1])


featName = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode', 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']

print('\n---- FEATURES ----\n')
print('atomic properties:', featName)

columns = {name: i for i, name in enumerate(featName)}

ids = {}
affinity = {}
coords = {}
features = {}

for dictionary in [ids, affinity, coords, features]:
    for dataset_name in datasets:
        dictionary[dataset_name] = []

NaN_present = ['2haw', '2o1c', '4lv1', '4mnv']
for dataset_name in datasets:
    pocket_dataset_path = os.path.join(args.input_dir, '%s_set_pocket.hdf' % dataset_name)
    ligand_dataset_path = os.path.join(args.input_dir, '%s_set_ligand.hdf' % dataset_name)
    with h5py.File(pocket_dataset_path, 'r') as f_p, \
          h5py.File(ligand_dataset_path, 'r') as f_l:
        for pdb_id in f_l:
            pocket_dataset = f_p[pdb_id]
            ligand_dataset = np.array(f_l[pdb_id])

            NAN_check = False

            # if(torch.isnan(torch.Tensor(pocket_dataset)).any() or torch.isnan(torch.Tensor(ligand_dataset)).any()):
            #   print(pdb_id)


            # for i in pocket_dataset:

            #   if(True in list(np.isnan(i))):
            #     NAN_check = True
            #     print(pdb_id)
            # if(True in list(np.isnan(ligand_dataset))):
            #   NAN_check = True
            #   print(pdb_id)
            if(pdb_id not in NaN_present):
              coords[dataset_name].append(pocket_dataset[:, :3])
              features[dataset_name].append({'pocket': pocket_dataset[:, 3:], 'ligand':ligand_dataset})
              affinity[dataset_name].append(pocket_dataset.attrs['affinity'])
              ids[dataset_name].append(pdb_id)

    ids[dataset_name] = np.array(ids[dataset_name])
    print(ids[dataset_name].shape)
    affinity[dataset_name] = np.reshape(affinity[dataset_name], (-1, 1))

for dataset_name in datasets:
    for i in range(len(affinity[dataset_name])):
        if(affinity[dataset_name][i][0].shape == (2,)):
            affinity[dataset_name][i][0] = affinity[dataset_name][i][0][1]


# normalize charges
charges = []
for feature_data in features['training']:
    charges.append(feature_data['pocket'][..., columns['partialcharge']])

charges = np.concatenate([c.flatten() for c in charges])

m = charges.mean()
std = charges.std()
print('charges: mean=%s, sd=%s' % (m, std))
print('use sd as scaling factor')

print('\n---- DATA ----\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('==> Building network..')
net = netpdb()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net,  device_ids = list(range(torch.cuda.device_count()))[:1])

criterion = nn.MSELoss()
net = net.to(device)

start_epoch, start_step = 0, 0
       
if args.resume:
    if(os.path.isfile('../save/network.ckpt')):
        net.load_state_dict(torch.load('../save/network.ckpt'))
        print("=> Network : loaded")
    
    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            start_epoch, start_step = (int(i) for i in str(f.read()).split(" "))
print("=> Network : prev epoch found")

def train(ID, epoch, coords, features, affinity, rot, std, lr = 1e-5):
  trainset = DB(coords, features, affinity, 'training', rot, std)
  dataloader = torch.utils.data.DataLoader(trainset, batch_size=28, shuffle=True)
  dataloader = iter(dataloader)
  print('\nID : %d | Epoch: %d | Rotations: %d ' % (ID, epoch, rot))

  train_loss, correct, total = 0, 0, 0
  params = net.parameters()

  if(ID in [0, 3, 4]):
    optimizer = optim.Adam(params, lr =lr)#, momentum=0.9)#, weight_decay=5e-4)
  elif(ID in [1, 2]):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-6, max_lr = 1e-3, mode = 'exp_range')
  
  for batch_idx in range(len(dataloader)):

    inputs_pocket, inputs_ligand, targets = next(dataloader)
    inputs_pocket, inputs_ligand, targets = inputs_pocket.to(device), inputs_ligand.to(device), targets.to(device)

    optimizer.zero_grad()
    y_pred = net(inputs_pocket, inputs_ligand)
    loss = criterion(y_pred, targets)
    loss.backward()
    
    if ID in [0, 3, 4]:
      optimizer.step()
    elif ID in [1, 2]:
      optimizer.step()
      scheduler.step()

    train_loss += loss.item()
    # NOTE : Logging here
    total += targets.size(0)

    with open("../save/logs/train_loss_%d.log"%(ID), "a+") as lfile:
      lfile.write("{}\n".format(train_loss / total))

    del inputs_pocket, inputs_ligand, targets
    gc.collect()
    torch.cuda.empty_cache()

    

    with open("../save/info_%d.txt"%ID, "w+") as f:
      f.write("{} {}".format(epoch, batch_idx))

    progress_bar(batch_idx, len(dataloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1)))
  torch.save(net.state_dict(), '../save/%d-network-%d.ckpt'%(ID, epoch))

  print(train_loss)

def test(dataset_name, coords, features, affinity, std, rot = 0):
  testset = DB(coords, features, affinity, dataset_name, rot, std)
  dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
  dataloader = iter(dataloader)
  print('\n%s: %d | Rotations: %d ' % (dataset_name, 1, rot))

  val_loss = 0
  inputs_pocket, inputs_ligand, targets = next(dataloader)
  inputs_pocket, inputs_ligand, targets = inputs_pocket.to(device), inputs_ligand.to(device), targets.to(device)
  y_pred = net(inputs_pocket, inputs_ligand)
  loss = criterion(y_pred, targets)
  val_loss += loss.item()

  print("validation loss : ", val_loss)

def store_results(ID,epoch, coords, features, affinity, std, rot = 0):

  predictions = []
  for dataset in ['training', 'validation', 'test']:
    testset = DB(coords, features, affinity, dataset, rot, std)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


    for batch_idx in range(len(dataloader)):
      dataloader = iter(dataloader)
      inputs_pocket, inputs_ligand, targets = next(dataloader)
      inputs_pocket, inputs_ligand, targets = inputs_pocket.to(device), inputs_ligand.to(device), targets.to(device)
      pred = net(inputs_pocket, inputs_ligand)
      pred = pred.detach().cpu().numpy()
      
      # print(ids[dataset].shape, affinity[dataset].shape, pred)
      predictions.append(pd.DataFrame(data={'pdbid': ids[dataset][batch_idx],
                            'real': affinity[dataset][batch_idx, 0],
                            'predicted': pred[0, 0],
                            'set': dataset}, index = [0]))

  predictions = pd.concat(predictions, ignore_index=True)
  predictions.to_csv('../save/csv/'+str(ID) + '_' + str(epoch) + '-predictions.csv', index=False)


for i in range(start_epoch, args.num_epochs):
  for j in (args.rotations):
    
    train(1,i ,coords, features, affinity, j, std, lr= 1e-5)
    test('validation', coords, features, affinity, std)
  store_results(1,i, coords, features, affinity, std)

test('test', coords, features, affinity, std)

for i in range(start_epoch, args.num_epochs):
  for j in (args.rotations):
    
    train(3,i ,coords, features, affinity, j, std, lr= 1e-6)
    test('validation', coords, features, affinity, std)
  store_results(3,i, coords, features, affinity, std)

test('test', coords, features, affinity, std)


print('==> Building network..')
net = net()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net,  device_ids = list(range(torch.cuda.device_count()))[:1])

criterion = nn.MSELoss()
net = net.to(device)

start_epoch, start_step = 0, 0
for i in range(start_epoch, args.num_epochs):
  for j in (args.rotations):
    
    train(2,i ,coords, features, affinity, j, std, lr= 1e-5)
    test('validation', coords, features, affinity, std)
  store_results(2,i, coords, features, affinity, std)

test('test', coords, features, affinity, std)

for i in range(start_epoch, args.num_epochs):
  for j in (args.rotations):
    
    train(4,i ,coords, features, affinity, j, std, lr= 1e-6)
    test('validation', coords, features, affinity, std)
  store_results(4,i, coords, features, affinity, std)

test('test', coords, features, affinity, std)
