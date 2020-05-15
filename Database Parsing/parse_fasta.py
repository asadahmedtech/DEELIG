PATH = '/home/binnu/Asad/dataset/new_db/fasta_single.txt'
OUT_PATH = '/home/binnu/Asad/dataset/new_db/fasta_single_formatted.txt'

file = open(PATH, "r")
data = file.readlines()
file.close()

temp_data, deleted_data = [], []

for i in range(len(data)):
	if(data[i][0] == ">"):
		ID = data[i]
		temp_data.append(data[i])
		count = 0
	else:
		if(len(data[i]) < 50 and data[i-1][0] == ">"):
			deleted_data.append(data[i])
			temp_data.pop()
		else:
			temp_data.append(data[i])
with open(OUT_PATH, "w") as f:
	f.writelines(temp_data)

print("==> Original Data : ", len(data))
print("==> Formatted Data : ", len(temp_data))
print("==> Deleted Data : ", deleted_data	, len(deleted_data))

