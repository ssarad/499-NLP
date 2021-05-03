#Data Extraction

file_train_indmnTknzd = "data/tokenized/in_domain_train.tsv"

def getLinesFromFile(fname):
	""" Given a text file, extract each line and put it into a list """
	with open(fname, encoding="utf-8") as inp:
		lines = inp.readlines()
	return lines

def extractFromDataset(datasetLines):
	""" Extract information from the given list of strings, which will be in a four-column format, with the columns separated by tab characters """
	# As stated on the CoLa website, the four columns hold the following informaiton:
	#  Column 1: the code representing the source of the sentence.
	#  Column 2: the acceptability judgment label (0=unacceptable, 1=acceptable).
	#  Column 3: the acceptability judgment as originally notated by the author.
	#  Column 4: the sentence.
	lineCtr = 0
	for line in datasetLines:
		datasetLines[lineCtr] = line.split("\t") # Split into 4 parts
		datasetLines[lineCtr][3] = datasetLines[lineCtr][3][:-1] # Remove trailing newline character from sentence
		lineCtr += 1
	return datasetLines # The lines have been broken apart as follows:
	                    # [source, acceptability label, acceptability label (original), sentence (with trailing newline removed)]

# Test line is below
print(f"Results: {extractFromDataset((getLinesFromFile(file_train_indmnTknzd))[:3])}")