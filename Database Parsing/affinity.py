import urllib.request
import re
import os
from bs4 import BeautifulSoup

import xlsxwriter 
import math

workbook = xlsxwriter.Workbook('affinity.xlsx') 
worksheet = workbook.add_worksheet("Sheet1") 

base_url = 'http://www.rcsb.org/structure/'

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

for filename in ["gen", "ref"]:
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [i.split(" ") for i in lines]        
        for i in lines:
            while('' in i):
                i.remove('')
        
        
        for i in lines:
            if(len(i)>5):
                if(str(i[0]) not in exce):
                    exce.append(str(i[0]))
                    worksheet.write(row_sheet, 0, str(i[0]))
                    worksheet.write(row_sheet, 1, i[3])
                    row_sheet += 1
        


workbook.close()

