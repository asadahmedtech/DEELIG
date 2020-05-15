
# Program extracting first column 
import xlsxwriter 
import xlrd 
  
wb = xlrd.open_workbook("affinity.xlsx") 
sheet = wb.sheet_by_index(0) 
sheet.cell_value(0, 0) 
  
lig_ID, affin, ligands = [], [], []
for i in range(sheet.nrows): 
    # if(len(sheet.cell_value(i, 1)) < 3):
    # 	PDB_ID.append(sheet.cell_value(i,0))
    # 	wrong_lig.append(sheet.cell_value(i, 1))
    ligands.append(sheet.cell_value(i, 1)) 
    lig_ID.append(sheet.cell_value(i,0).upper()+"_"+sheet.cell_value(i, 1))
    affin.append(sheet.cell_value(i, 2))

unique_lig = []
lig_dict = {}
for i in ligands:
	if(i not in unique_lig):
		lig_dict[i] = 1
		unique_lig.append(i)
	else:
		lig_dict[i] += 1

print("Total Number of ligands : ", len(ligands))
print("Unique Number of Ligands : ", len(unique_lig))
print(lig_ID)
# print(wrong_lig)
# workbook = xlsxwriter.Workbook('PDB_ligands_ID.xlsx') 
# worksheet = workbook.add_worksheet("Sheet1") 

# row = 0

# for i in unique_lig:
# 	worksheet.write(row, 0, i)
# 	worksheet.write(row, 1, lig_dict[i])
# 	row += 1

# workbook.close()

# with open("PDB_ligands_ID.txt", "w") as f:
# 	f.writelines(lig_ID)

# workbook = xlsxwriter.Workbook('PDB_ligands_affinity.xlsx') 
# worksheet = workbook.add_worksheet("Sheet1") 

# row = 0

# for i in range(len(lig_ID)):
# 	worksheet.write(row, 0, lig_ID[i])
# 	worksheet.write(row, 1, affin[i])
# 	row += 1

# workbook.close()