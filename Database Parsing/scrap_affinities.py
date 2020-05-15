import urllib.request
import re
import os
from bs4 import BeautifulSoup

import xlsxwriter 
import math

workbook = xlsxwriter.Workbook('affinity.xlsx') 
worksheet = workbook.add_worksheet("Sheet1") 

base_url = 'http://www.rcsb.org/structure/'

path = '/home/binnu/Asad/dataset/new_db/protein_pdb/'
files = os.listdir(path)

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def convert(affinity, unit):
	if(affinity == 0.0):
		return 0
	print(affinity)
	if(unit == 'nM'):
		return round_up(-math.log(affinity * 10e-9), 2)
	elif(unit == 'pM'):
		return round_up(-math.log(affinity * 10e-12), 2)
	elif(unit == 'mM'):
		return round_up(-math.log(affinity * 10e-3), 2)
	elif(unit == 'fM'):
		return round_up(-math.log(affinity * 10e-15), 2)
	elif(unit == 'uM'):
		return round_up(-math.log(affinity * 10e-6), 2)
	else:
		return 0

no_kd, kd_1, kd_2, kd_3 = 0, 0, 0, 0
exce = []
row_sheet = 0

for file in files:
	if(file.endswith("pdb")):
		pdb_id = file[:-4]

		req = urllib.request.Request("https://www.rcsb.org/structure/" + pdb_id)
		print(pdb_id)

		resp = urllib.request.urlopen(req)
		respData = resp.read()
		soup = BeautifulSoup(str(respData), features = "lxml")

		table = soup.find('table', attrs={'id':'ExternalLigandAnnotationMainTable'})
		
		if(table != None):
			table_body = table.find('tbody')

			rows = table_body.find_all('tr')
			data = []
			for row in rows:
			    cols = row.find_all('td')
			    cols = [ele.text.strip() for ele in cols]
			    data.append([ele for ele in cols if ele]) # Get rid of empty values
			for row in data:
				try:
					row[1] = row[1].replace("&nbsp", " ")
					row[1] = row[1].replace("(", " ")
				except Exception as e:
					pass

				row[1] = row[1].split()
				# print(row)
				if("BIND" in row[1][-1]):
					row[1].pop(-1)
				if(")" in row[1][-1]):
					row[1].pop(-1)

				# print(row)
				# row[1] = row[1].replace("(100)&nbspBINDINGDB", '')
				# row[1] = row[1].replace("&nbspBINDINGMOAD", '')
				# row[1] = row[1].replace("&nbspPDBBIND", '')
				# row[1] = row[1].split()
				row[1][0] = row[1][0][:-1]
				if(len(row[1]) == 5):
					row[1].pop(1)
					row[1].pop(1)
				if(row[1][1][0] in ['~', '>', '<']):
					row[1][1] = row[1][1][1:]
			
			ligands = []
			temp_data = {}
			for d in data:
				if(d[0] not in ligands):
					ligands.append(d[0])
					temp_data[d[0]] = d
				else:
					if(temp_data[d[0]][1][0] == "IC50"):
						temp_data[d[0]] = d 
			data = list(temp_data.values())
			for d in data:
				worksheet.write(row_sheet, 0, str(pdb_id))
				worksheet.write(row_sheet, 1, d[0])
				worksheet.write(row_sheet, 2, convert(float(d[1][1]), d[1][2]))
				worksheet.write(row_sheet, 3, d[1][1] + " " + d[1][2])
				worksheet.write(row_sheet, 4, d[1][0])
				row_sheet += 1

			print(data)
		else:
			exce.append(pdb_id)
			print("No Binding Affinity Found")

workbook.close()

with open("free_ligands.txt", "w") as f:
	f.writelines(exce)
