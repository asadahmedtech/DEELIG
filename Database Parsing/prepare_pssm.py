import os
pssm_path = '/home/binnu/Asad/dataset/new_db/pssm/'
fasta_path = '/home/binnu/Asad/dataset/new_db/fasta_single_formatted.txt'

with open(fasta_path, 'r') as file:
	fasta_seq = file.readlines()

order = []
for i in fasta_seq:
	if(i[0] =='>'):
		temp = i[1:5] + '_' + i[6]
		order.append(temp)
start = 0
# start = 5461
errors = []
for i in range(start, len(order)):
	file = pssm_path + 'outpssm_' + str(i+1)
	try:
		os.rename(file, pssm_path + order[i])
	except:
		errors.append(i)
		pass
print(errors)