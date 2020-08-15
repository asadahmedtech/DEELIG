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

errors = []
naccess_errors = []

def parse_PSSM(file, path = '/home/binnu/Asad/dataset/pdbbind/pssm/'):
	pssm = {}
	file = "outpssm_" + file
	try:
		with open(os.path.join(path, file), 'r') as f:
			lines = f.readlines()
			# lines = [i.split() if(len(i.split()) == 44) for i in lines]
			lines_new = []
			for i in lines:
				i = i.split()
				if(len(i) == 44):
					lines_new.append(i)
			lines_new = [i[:22] for i in lines_new]
	except Exception as e:
		print("NOT PRESENT: ", file)
		return None

	for i in lines_new:
		scores = i[2:]
		scores = [int(temp_i) for temp_i in scores]
		pssm[i[0]] = scores
	# print(pssm)
	return pssm
	
def calc_features(PATH, pdb_ligand_ID, OUTPATH):

	#Loading the files
	global errors, naccess_errors
	parser = PDBParser(PERMISSIVE = 1)

	PDB_id = pdb_ligand_ID[:4].lower() #+ '_pocket'
	filename = os.path.join(PATH, PDB_id + "_protein.pdb")
	try:
		structure = parser.get_structure(PDB_id, filename)
	except Exception as e:
		print("FILE NOT PRESENT: ", PDB_id)
		errors.append(PDB_id)
		return
	model = structure[0]

	#DSSP Analysis for SS, PHI, PSI
	try:
		dssp = DSSP(model, filename)
	except Exception as e:
		naccess_errors.append(PDB_id)
		print("NACCESS NOT PRESENT: ", PDB_id)
		return 

	#NACCESS Analysis for SASA
	try:
		rsa, asa = run_naccess(model, filename)
		rsa = process_rsa_data(rsa)
	except Exception as e:
		naccess_errors.append(PDB_id)
		print("NACCESS NOT PRESENT: ", PDB_id)
		return 
	# print(rsa)
	#Feature mapping to each atomic coordinate
	dssp_present, dssp_not_present = 0, 0
	
	feature = dict() #The feature dictionary

	try:
		for model in structure:
			for chain in model:
				# Modify Output PDB to compute all the chains in a single complex.
				pssm_ID = chain.get_full_id()[0][:4].upper() + '_' + str(chain.get_full_id()[1]+1)
				pssm = parse_PSSM(pssm_ID)
				if pssm == None:
					continue
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
	except Exception as e:
		print("FILE NOT PRESENT: ", PDB_id)
		errors.append(PDB_id)
		return
	#Printing the Stats 
	print("===> STATS : PDBID : %s , DSSP PRESENT : %s , DSSP NOT PRESENT : %s"%(PDB_id, dssp_present, dssp_not_present))

	#Saving the feature to each PDB file
	with open(os.path.join(OUTPATH, pdb_ligand_ID + ".dat"), "wb+") as f:
		pickle.dump(feature, f)
		print("====> Dump completed")


if __name__ == '__main__':
	global errors, naccess_errors

	input_dir = '/home/binnu/Asad/dataset/pdbbind/protein_pdb/'
	output_dir = "/home/binnu/Asad/dataset/pdbbind/protein_pdb_featurized/"
	IDs = '/home/binnu/Asad/dataset/pdbbind/PDBBind.txt'

	# Loading the IDs and removing the next line charecter.
	files = open(IDs, 'r')
	files = files.readlines()
	files = [i[:-1] for i in files]

	# Listing the files that are already computed.
	files_done = os.listdir(output_dir)
	files_done = [i[:-4] for i in files_done]
	print(files_done)

	not_present_files = open("not_present.txt", 'r')
	not_present_files = not_present_files.readlines()
	not_present_files = [i[:-1] for i in not_present_files]

	naccess_not_present_files = open("nacces_not_present.txt", 'r')
	naccess_not_present_files = naccess_not_present_files.readlines()
	naccess_not_present_files = [i[:-1] for i in naccess_not_present_files]


	# Add the PDB IDs of file which NACCESS gives segmentation fault.
	naccess_error = ['5FQD_LVY', '4EJG_NCT', '3N7A_FA1' , '2IJ7_TPF', '4EJH_0QA','2QJY_SMA','1WPG_TG1', '2A06_SMA','4UHL_VFV','3N8K_D1X','5FV9_Y6W','3N75_G4P','3B8H_NAD','3B82_NAD','3B78_NAD']
	for file in files:
		if(file not in files_done) and file not in naccess_error and file.lower() not in not_present_files and file.lower() not in naccess_not_present_files:
			print("==> Featurizing : ", file)
			calc_features(input_dir, file, output_dir)

		with open("not_present.txt", "w") as f:
			temp = [i+"\n" for i in errors]
			f.writelines(temp)
			f.flush()
			f.close()
		with open("nacces_not_present.txt", "w") as f:
			temp = [i+"\n" for i in naccess_errors]
			f.writelines(temp)
			f.flush()
			f.close()