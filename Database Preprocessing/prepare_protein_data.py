from Bio import *
import os

#PDB Parser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.NACCESS import run_naccess, process_rsa_data
import pickle 

#Labels for files
"""Secondary Structure in Millers Format
	H - Alpha Helix (4-12)            = 1
	B - Isolated Beta Bridge residue  = 2	
	E - Strand                        = 3
	G - 3-10 helix                    = 4
	I - Pi Helix                      = 5
	T - Turn                          = 6
	S - bend                          = 7
	- - None                          = 0
"""
SS_Labels = {'H' : 1, 'B' : 2, 'E' : 3, 'G' : 4, 'I' : 5, 'T' : 6, 'S' : 7, '-' : 0}
"""Relactive Solvent Accessiblity (RSA)
	Threshold = 25
	Exposed (> Threshold) = 1
	Burried (<= Threshold) = 0
"""
RSA_Threshold = 25

def parse_PSSM(file, path = '/home/binnu/Asad/dataset/new_db/pssm/'):
	pssm = {}
	with open(os.path.join(path, file), 'r') as f:
		lines = f.readlines()
		# lines = [i.split() if(len(i.split()) == 44) for i in lines]
		lines_new = []
		for i in lines:
			i = i.split()
			if(len(i) == 44):
				lines_new.append(i)
		lines_new = [i[:22] for i in lines_new]
	
	for i in lines_new:
		scores = i[2:]
		scores = [int(temp_i) for temp_i in scores]
		pssm[i[0]] = scores
	# print(pssm)
	return pssm
	
def calc_features(PATH, pdb_ligand_ID, OUTPATH):

	#Loading the files
	parser = PDBParser(PERMISSIVE = 1)

	PDB_id = pdb_ligand_ID[:4].lower() #+ '_pocket'
	filename = os.path.join(PATH, PDB_id + ".pdb")
	structure = parser.get_structure(PDB_id, filename)
	model = structure[0]

	#DSSP Analysis for SS, PHI, PSI
	dssp = DSSP(model, filename)

	#NACCESS Analysis for SASA
	rsa, asa = run_naccess(model, filename)
	rsa = process_rsa_data(rsa)
	# print(rsa)
	#Feature mapping to each atomic coordinate
	dssp_present, dssp_not_present = 0, 0
	
	feature = dict() #The feature dictionary

	for model in structure:
		for chain in model:
			if(chain.get_full_id()[2] == pdb_ligand_ID.split('_')[2]):
				pssm_ID = chain.get_full_id()[0][:4].upper() + '_' + chain.get_full_id()[2]
				pssm = parse_PSSM(pssm_ID)
				start = True
				gap = 0
				idx_prev = 0
				for residue in chain:
					# if(start):
						# start_idx =residue.get_full_id()[3][1]
						# idx_prev = 0
					idx = residue.get_full_id()[3][1]
					if(idx < 1):
						print(idx)
						a = 0
						pass
					elif(idx - idx_prev >= 1):
						print(idx)
						a = 1
						gap += idx - idx_prev -1
					# elif(start):
						# gap += -1
						# start = False
					
					
						for atom in residue:
							# print(atom.get_full_id())					
							ID = (atom.get_full_id()[2], atom.get_full_id()[3])

							if(ID in list(dssp.keys())):
								if(rsa[ID]["all_atoms_abs"] > RSA_Threshold):
									rsa_label = 1
								else:
									rsa_label = 0

								print(gap, atom.get_full_id()[3][1], a)
								feat = (SS_Labels[dssp[ID][2]], dssp[ID][4]/360, dssp[ID][5]/360, rsa_label) + tuple(pssm[str(atom.get_full_id()[3][1] - gap)])
								feature[tuple(atom.get_coord())] = feat

								print(pdb_ligand_ID[:4], ID, atom.get_coord(), feat)
								dssp_present += 1

							else:
								print(">>> ID not present : ", atom.get_full_id())
								dssp_not_present += 1
						idx_prev = idx

	#Printing the Stats 
	print("===> STATS : PDBID : %s , DSSP PRESENT : %s , DSSP NOT PRESENT : %s"%(PDB_id, dssp_present, dssp_not_present))

	#Saving the feature to each PDB file
	with open(os.path.join(OUTPATH, pdb_ligand_ID + ".dat"), "wb+") as f:
		pickle.dump(feature, f)
		print("====> Dump completed")


if __name__ == '__main__':
	input_dir = '/home/binnu/Asad/dataset/new_db/protein_pdb/'
	output_dir = "/home/binnu/Asad/dataset/new_db/protein_pdb_featurized/"
	IDs = '/home/binnu/Asad/dataset/new_db/PDB_ligands_chain_ID_pssm_Admet_padel.txt'

	files = open(IDs, 'r')
	files = files.readlines()
	files = [i[:-1] for i in files]

	files_done = os.listdir(output_dir)
	files_done = [i[:-4] for i in files_done]
	print(files_done)

	naccess_error = ['5FQD_LVY', '4EJG_NCT', '3N7A_FA1' , '2IJ7_TPF', '4EJH_0QA','2QJY_SMA','1WPG_TG1', '2A06_SMA','4UHL_VFV','3N8K_D1X','5FV9_Y6W','3N75_G4P','3B8H_NAD','3B82_NAD','3B78_NAD']
	for file in files:
		if(file not in files_done) and file.split('_')[0] + '_' + file.split('_')[1] not in naccess_error:
			print("==> Featurizing : ", file)
			calc_features(input_dir, file, output_dir)
