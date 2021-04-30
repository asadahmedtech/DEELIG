from sklearn.utils import shuffle
import h5py

import os

import argparse

parser = argparse.ArgumentParser(description='Split dataset int training,'
                                 ' validation and test sets.')
parser.add_argument('--input_path', '-i', default='../../processed_data_pdbbind',
                    help='directory with pdbbind dataset')
parser.add_argument('--output_path', '-o', default='../../dataset_pdbbind',
                    help='directory to store output files')
parser.add_argument('--size_val', '-s', type=int, default=500,
                    help='number of samples in the validation set')
parser.add_argument('--size_test', '-t', type=int, default=500,
                    help='number of samples in the validation set')
args = parser.parse_args()

casf13 = open('casf2013.txt', 'r')
casf13 = casf13.readlines()
casf13 = [i[:4].upper() for i in casf13]
print(casf13[0])

casf16 = open('casf2016.txt', 'r')
casf16 = casf16.readlines()
casf16 = [i[:4].upper() for i in casf16]
print(casf16[0])

casf = set()
for a in casf13:
    casf.add(a)
for a in casf16:
    casf.add(a)

casf = list(casf)
print(len(casf))

casfinVal = 0
casfinTrain = []
casftoMove = 0
casfinValtoTrain = []
# create files with the training and validation sets
with h5py.File('%s/training_set_pocket.hdf' % args.output_path, 'w') as g_p, \
     h5py.File('%s/validation_set_pocket.hdf' % args.output_path, 'w') as h_p, \
     h5py.File('%s/test_set_pocket.hdf' % args.output_path, 'w') as i_p, \
     h5py.File('%s/training_set_ligand.hdf' % args.output_path, 'w') as g_l, \
     h5py.File('%s/validation_set_ligand.hdf' % args.output_path, 'w') as h_l, \
     h5py.File('%s/test_set_ligand.hdf' % args.output_path, 'w') as i_l:

    with h5py.File('%s/training_set_pocket.hdf' % args.input_path, 'r') as g_p_old, \
     h5py.File('%s/validation_set_pocket.hdf' % args.input_path, 'r') as h_p_old, \
     h5py.File('%s/test_set_pocket.hdf' % args.input_path, 'r') as i_p_old, \
     h5py.File('%s/training_set_ligand.hdf' % args.input_path, 'r') as g_l_old, \
     h5py.File('%s/validation_set_ligand.hdf' % args.input_path, 'r') as h_l_old, \
     h5py.File('%s/test_set_ligand.hdf' % args.input_path, 'r') as i_l_old:
        
        #PDBBind Set
        print("PDBBind Before: ", len(list(f_p.keys())), len(list(f_l.keys())), casfinVal)
        for pdb_id in list(g_p_old.keys()):
            if(pdb_id[:4].upper() in casf):
                casfinTrain.append(pdb_id)
        
        for pdb_id in list(h_p_old.keys()):
            if(casftoMove>=len(casfinTrain)):
                break
            if(pdb_id[:4].upper() not in casf):
                casfinValtoTrain.append(pdb_id)
                casftoMove += 1

        print("CASF in Train: {}".format(casftoMove))
        
        for pdb_id in list(g_p_old.keys()):
            if(pdb_id not in casfinTrain):
                ds_p = g_p.create_dataset(pdb_id, data=g_p_old[pdb_id])
                ds_p.attrs['affinity'] = g_p_old[pdb_id].attrs['affinity']
                ds_l = g_l.create_dataset(pdb_id, data=g_l_old[pdb_id])
                ds_l.attrs['affinity'] = g_l_old[pdb_id].attrs['affinity']
                
        for pdb_id in list(g_p_old.keys()):
            if(pdb_id in casfinTrain):
                ds_p = h_p.create_dataset(pdb_id, data=g_p_old[pdb_id])
                ds_p.attrs['affinity'] = g_p_old[pdb_id].attrs['affinity']
                ds_l = h_l.create_dataset(pdb_id, data=g_l_old[pdb_id])
                ds_l.attrs['affinity'] = g_l_old[pdb_id].attrs['affinity']

        for pdb_id in list(h_p_old.keys()):
            if(pdb_id not in casfinValtoTrain):
                ds_p = g_p.create_dataset(pdb_id, data=h_p_old[pdb_id])
                ds_p.attrs['affinity'] = h_p_old[pdb_id].attrs['affinity']
                ds_l = g_l.create_dataset(pdb_id, data=h_l_old[pdb_id])
                ds_l.attrs['affinity'] = h_l_old[pdb_id].attrs['affinity']
                
        for pdb_id in list(h_p_old.keys()):
            if(pdb_id in casfinValtoTrain):
                ds_p = h_p.create_dataset(pdb_id, data=g_p_old[pdb_id])
                ds_p.attrs['affinity'] = g_p_old[pdb_id].attrs['affinity']
                ds_l = h_l.create_dataset(pdb_id, data=g_l_old[pdb_id])
                ds_l.attrs['affinity'] = g_l_old[pdb_id].attrs['affinity']
        
        for pdb_id in list(i_p_old.keys()):
            ds_p = i_p.create_dataset(pdb_id, data=i_p_old[pdb_id])
            ds_p.attrs['affinity'] = i_p_old[pdb_id].attrs['affinity']
            ds_l = i_l.create_dataset(pdb_id, data=i_l_old[pdb_id])
            ds_l.attrs['affinity'] = i_l_old[pdb_id].attrs['affinity']

        print("Train : {} :: Test : {} :: Val : {} ".format(len(list(g_p.keys())), len(list(h_p.keys())), len(list(i_p.keys()))))