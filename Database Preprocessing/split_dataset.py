from sklearn.utils import shuffle
import h5py

import os

import argparse

parser = argparse.ArgumentParser(description='Split dataset int training,'
                                 ' validation and test sets.')
parser.add_argument('--input_path', '-i', default='../../processed_data',
                    help='directory with pdbbind dataset')
parser.add_argument('--output_path', '-o', default='../../dataset',
                    help='directory to store output files')
parser.add_argument('--size_val', '-s', type=int, default=500,
                    help='number of samples in the validation set')
parser.add_argument('--size_test', '-t', type=int, default=200,
                    help='number of samples in the validation set')
args = parser.parse_args()

# create files with the training and validation sets
with h5py.File('%s/training_set_pocket.hdf' % args.output_path, 'w') as g_p, \
     h5py.File('%s/validation_set_pocket.hdf' % args.output_path, 'w') as h_p, \
     h5py.File('%s/test_set_pocket.hdf' % args.output_path, 'w') as i_p, \
     h5py.File('%s/training_set_ligand.hdf' % args.output_path, 'w') as g_l, \
     h5py.File('%s/validation_set_ligand.hdf' % args.output_path, 'w') as h_l, \
     h5py.File('%s/test_set_ligand.hdf' % args.output_path, 'w') as i_l:

    with h5py.File('%s/data_pocket.hdf' % args.input_path, 'r') as f_p, \
        h5py.File('%s/data_ligand.hdf' % args.input_path, 'r') as f_l:
        data_shuffled = shuffle(list(f_p.keys()), random_state=123)
        print(len(list(f_l.keys())))
        f_l_keys = list(f_l.keys())
        for pdb_id in data_shuffled[:args.size_val]:
            if(pdb_id in f_l_keys):
                ds_p = h_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = h_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']]
            else:
                print(pdb_id)
        for pdb_id in data_shuffled[args.size_val:args.size_test+args.size_val]:
            if(pdb_id in f_l_keys):
                ds_p = i_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = i_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']
            else:
                print(pdb_id)
        for pdb_id in data_shuffled[args.size_test+args.size_val:] :
            if(pdb_id in f_l_keys):
                ds_p = g_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = g_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']
            else:
                print(pdb_id)
