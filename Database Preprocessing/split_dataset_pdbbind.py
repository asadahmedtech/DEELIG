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
# create files with the training and validation sets
with h5py.File('%s/training_set_pocket.hdf' % args.output_path, 'w') as g_p, \
     h5py.File('%s/validation_set_pocket.hdf' % args.output_path, 'w') as h_p, \
     h5py.File('%s/test_set_pocket.hdf' % args.output_path, 'w') as i_p, \
     h5py.File('%s/training_set_ligand.hdf' % args.output_path, 'w') as g_l, \
     h5py.File('%s/validation_set_ligand.hdf' % args.output_path, 'w') as h_l, \
     h5py.File('%s/test_set_ligand.hdf' % args.output_path, 'w') as i_l:

    with h5py.File('%s/data_pocket.hdf' % args.input_path, 'r') as f_p, \
        h5py.File('%s/data_ligand.hdf' % args.input_path, 'r') as f_l:
        
        #PDBBind Set
        print("PDBBind Before: ", len(list(f_p.keys())), len(list(f_l.keys())), casfinVal)
        for pdb_id in list(f_p.keys()):
            if(pdb_id[:4].upper() in casf and pdb_id in list(f_l.keys())):
                casfinVal += 1
                ds_p = h_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = h_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']

                del f_p[pdb_id]
                del f_l[pdb_id]
            else:
                print(pdb_id)

        print("PDBBind After: ", len(list(f_p.keys())), len(list(f_l.keys())), casfinVal)

        data_shuffled = shuffle(list(f_p.keys()), random_state=123)
        print(len(list(f_l.keys())))
        f_l_keys = list(f_l.keys())

        for pdb_id in data_shuffled[:args.size_val-casfinVal]:
            if(pdb_id in f_l_keys):
                ds_p = h_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = h_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']
            else:
                print(pdb_id)
        for pdb_id in data_shuffled[args.size_val-casfinVal:args.size_test+args.size_val-casfinVal]:
            if(pdb_id in f_l_keys):
                ds_p = i_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = i_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']
            else:
                print(pdb_id)
        for pdb_id in data_shuffled[args.size_test+args.size_val-casfinVal:] :
            if(pdb_id in f_l_keys):
                ds_p = g_p.create_dataset(pdb_id, data=f_p[pdb_id])
                ds_p.attrs['affinity'] = f_p[pdb_id].attrs['affinity']
                ds_l = g_l.create_dataset(pdb_id, data=f_l[pdb_id])
                ds_l.attrs['affinity'] = f_l[pdb_id].attrs['affinity']
            else:
                print(pdb_id)
