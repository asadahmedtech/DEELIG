import numpy as np
import pandas as pd
import h5py

import pybel
from extract_features import Featurizer

import csv
import os

ligand_PADEL_error = []
ligand_ADMET_error = []
ligand_404 = []
pocket_404 = []
def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path


def string_bool(s):
    s = s.lower()
    if s in ['true', 't', '1', 'yes', 'y']:
        return True
    elif s in ['false', 'f', '0', 'no', 'n']:
        return False
    else:
        raise IOError('%s cannot be interpreted as a boolean' % s)

def create_ligand_features(ID, ligand_featurefile_PADEL, ligand_featurefile_ADMET):
    # ID = '\"' + ID + '\"'
    ID = ID + "_ligand"
    try:
        feat1 = list(ligand_featurefile_PADEL.loc[ID])
    except Exception as e:
        try:
            ID = ID.split("_")[0].upper() + "_ligand"
            feat1 = list(ligand_featurefile_PADEL.loc[ID])
        except Exception as e:
            ligand_PADEL_error.append(ID.split("_")[0]+"\n")
            return []

    try:
        feat2 = list(ligand_featurefile_ADMET[ID])
    except Exception as e:
        try:
            ID = ID.split("_")[0].upper() + "_ligand"
            feat2 = list(ligand_featurefile_ADMET[ID])
        except Exception as e:
            ligand_ADMET_error.append(ID.split("_")[0]+"\n")
            return []


    PADEL_INDEX = [2, 40, 41, 3] + list(range(13, 22)) + list(range(2167, 13613))
    ADMET_INDEX = [1, 2, 3, 4, 5, 6,
                    10, 11, 12, 13, 14,
                    15, 16, 17, 21, 22, 
                    23, 24, 25, 26, 27, 
                    33, 34, 35, 39, 40,
                    41, 42, 43, 44, 45, 
                    46, 47, 48, 49, 38, 20]

    feat1 = [feat1[i] for i in PADEL_INDEX]
    feat2 = [feat2[i] for i in ADMET_INDEX]
    #print(feat1, feat2) 
    feat = feat1 + feat2
    #print(feat)
    try:
        d = np.array(feat, dtype = np.float32)
    except Exception:
        return []
    return d

def create_features(pocket_dir, ligand_dir, ID,  datafile_ligand, affinities, ligand_featurefile_PADEL, ligand_featurefile_ADMET, pocket_format = "pdb", ligand_format = "mol2"):
    
    pocket = next(pybel.readfile(pocket_format, os.path.join(pocket_dir, ID + "_pocket.%s"%(pocket_format))))
    try:
        ligand = next(pybel.readfile(ligand_format, os.path.join(ligand_dir, ID + '_ligand.mol2')))
    except Exception as e:
        ligand_404.append(ID)
        return
    try:        
        pocket_coords, pocket_features = featurizer.get_features(pocket, ID ,molcode=-1)
    except Exception as e:
       print("EOF ERROR ON : ", ID)
       pocket_404.append(ID)
       return
    ligand_coords, ligand_features = featurizer.get_features(ligand, None ,molcode=1)
        
    # ligand_features = create_ligand_features(ID, ligand_featurefile_PADEL, ligand_featurefile_ADMET)
    # print(ligand_features.shape)    
    centroid = ligand_coords.mean(axis=0)
    ligand_coords -= centroid
    pocket_coords -= centroid

    # data = np.concatenate(
    #     (np.concatenate((ligand_coords, pocket_coords)),
    #     np.concatenate((ligand_features, pocket_features))),
    #     axis=1,
    # )
    
    data_pocket = np.concatenate((pocket_coords, pocket_features), axis=1)
    
    dataset_pocket = datafile_pocket.create_dataset(ID, data=data_pocket, shape=data_pocket.shape,dtype='float32', compression='lzf')
    dataset_pocket.attrs['affinity'] = affinities.loc[ID]
    # if(ligand_features != []):
    #     print(ligand_features.shape)
    #     dataset_ligand = datafile_ligand.create_dataset(ID, data=ligand_features, shape=ligand_features.shape, dtype='float32')
    #     dataset_ligand.attrs['affinity'] = affinities.loc[ID]
    # 
    #     print("===> File dumped : ", ID)
    # else:
    #     print("===> Error : ", ID)
    #     error.append(ID)


if __name__ == '__main__':
    pocket_dir = '/home/binnu/Asad/dataset/pdbbind/pocket_pdb/'
    ligand_dir = '/home/binnu/Asad/dataset/pdbbind/ligand_mol2/'
    output_dir = "/home/binnu/Asad/dataset/pdbbind/processed_data/"
    protein_feature_path = '/home/binnu/Asad/dataset/pdbbind/protein_pdb_featurized/'
    featurefile_path_PADEL = '/home/binnu/Asad/dataset/pdbbind/ligand_PADEL.csv'
    featurefile_path_ADMET = '/home/binnu/Asad/dataset/pdbbind/ligand_ADMET.csv'
    affinities = '/home/binnu/Asad/dataset/pdbbind/affinity.csv'

    global featurizer, charge_idx
    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')
    
    # files = os.listdirs(pocket_dir)

    print("=> Loading Affinity csv")
    if(affinities is not None):
        affinities = pd.read_csv(affinities)
        if 'affinity' not in affinities.columns:
            raise ValueError('There is no `affinity` column in the table')
        elif 'name' not in affinities.columns:
            raise ValueError('There is no `name` column in the table')
        affinities = affinities.set_index('name')['affinity']
        print("=> Loaded Affinity csv")
    else:
        affinities = None

    # print("=> Loading PADEL csv")
    # ligand_featurefile_PADEL = pd.read_csv(featurefile_path_PADEL, index_col ='Name')
    # ligand_featurefile_ADMET = pd.read_csv(featurefile_path_ADMET, index_col = 'molecule')
    print("=> Loading ADMET csv")
    ligand_featurefile_ADMET = open(featurefile_path_ADMET, "r")
    ligand_ADMET_dic = dict()
    ligand_errors = []
    for lines in csv.reader(ligand_featurefile_ADMET, delimiter=","):
        if "molecule" in lines or '' in lines:
            ligand_errors.append(lines[0])
            # ligand_ADMET_dic[lines[0]] = [0 for i in range(52)]
            pass
        else:
            pass
            # ligand_ADMET_dic[lines[0]] = lines
    with open("ligand_errors_admet.txt","w") as f:
        ligand_errors = [i+"\n" for i in ligand_errors]
        f.writelines(ligand_errors)
        f.flush()
        f.close()
    print("=> Created ADMET dict")

    # for file in files:
    #     if(file.endswith(".pdb")):
    #         print("==> Creating Feature file : ", file)
    #         calc_features(pocket_dir, ligand_dir, file[:-4], output_dir)

    pdb_ligand_ID = '/home/binnu/Asad/dataset/pdbbind/PDBBind.txt'
    with open(pdb_ligand_ID, "r") as f:
        pdb_ligand_ID = f.readlines()
    for i in range(len(pdb_ligand_ID)):
        pdb_ligand_ID[i] = pdb_ligand_ID[i][:-1]
    iterr = 1
    print("=> Loading PDB IDs")

    
    global error
    error = []
    segmentation_fault = ['2QJY_SMA', '4DEL_PGH']
    once = True
    file_done = []
    with open('file_done.txt', 'r') as f:
        file_done = f.readlines()
        file_done = [i[:-1] for i in file_done]
    if(file_done != []):
       once = False
       
    not_present_files = open("not_present.txt", 'r')
    not_present_files = not_present_files.readlines()
    not_present_files = [i[:-1] for i in not_present_files]
    
    naccess_not_present_files = open("nacces_not_present.txt", 'r')
    naccess_not_present_files = naccess_not_present_files.readlines()
    naccess_not_present_files = [i[:-1] for i in naccess_not_present_files]
    
    naccess_error = ['5FQD_LVY', '4EJG_NCT', '3N7A_FA1' , '2IJ7_TPF', '4EJH_0QA','2QJY_SMA','1WPG_TG1', '2A06_SMA','4UHL_VFV','3N8K_D1X','5FV9_Y6W','3N75_G4P','3B8H_NAD','3B82_NAD','3B78_NAD']
    for ID in pdb_ligand_ID :
        if(once):
            datafile_pocket = h5py.File(os.path.join(output_dir, 'data_pocket.hdf'), "w")
            # datafile_ligand = h5py.File(os.path.join(output_dir, 'data_ligand.hdf'), "w")
            once = False
        else:
            datafile_pocket = h5py.File(os.path.join(output_dir, 'data_pocket.hdf'), "a")
            # datafile_ligand = h5py.File(os.path.join(output_dir, 'data_ligand.hdf'), "a")

        if(ID not in segmentation_fault and ID not in file_done and ID not in naccess_error and ID not in ligand_errors and ID not in not_present_files and ID not in naccess_not_present_files):
            print("==> Creating Feature file : ", ID, iterr)
            create_features(pocket_dir, ligand_dir, ID, datafile_pocket, affinities, None, None)
            file_done.append(ID + '\n')
            with open('file_done.txt', 'w') as f:
                f.writelines(file_done)
            iterr += 1 

        datafile_pocket.close()
        # datafile_ligand.close()
    
    with open("ligand_errors_padel.txt","w") as f:
        f.writelines(ligand_PADEL_error)
        f.flush()
        f.close()
    with open("ligand_errors_admet_run.txt","w") as f:
        f.writelines(ligand_ADMET_error)
        f.flush()
        f.close()

    with open("ligand_errors_mol2_run.txt","w") as f:
        ligand_404 = [i+"\n" for i in ligand_404]
        f.writelines(ligand_404)
        f.flush()
        f.close()
    with open("pocket404.txt","w") as f:
        ligand_404 = [i+"\n" for i in pocket_404]
        f.writelines(ligand_404)
        f.flush()
        f.close()

    print(error)
    # datafile_pocket = h5py.File(os.path.join(output_dir, 'data_pocket.hdf'), "w")
    # datafile_ligand = h5py.File(os.path.join(output_dir, 'data_ligand.hdf'), "w")
    # create_features(pocket_dir, ligand_dir, '5X67_UMP_B_401', datafile_pocket, datafile_ligand, affinities, ligand_featurefile_PADEL, ligand_featurefile_ADMET)
