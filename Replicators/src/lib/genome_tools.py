#%%
# https://en.wikipedia.org/wiki/DNA_and_RNA_codon_tables
GEN = {'00':'A', '01':'T', '10':'C', '11':'G'}
CODONS = {
'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
'AGA':'R', 'AGG':'R',
'AAT':'N', 'AAC':'N',
'GAT':'D', 'GAC':'D',
'TGT':'C', 'TGC':'C',
'CAA':'Q', 'CAG':'Q',
'GAA':'E', 'GAG':'E',
'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',
'CAT':'H', 'CAC':'H',
'ATT':'I', 'ATC':'I', 'ATA':'I',
'CTT':'C', 'CTC':'C', 'CTA':'C', 'CTG':'C', 'TTA':'C', 'TTG':'C',
'AAA':'K', 'AAG':'K',
'ATG':'M',
'TTT':'F', 'TTC':'F',
'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'AGT':'S', 'AGC':'S',
'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
'TGG':'W',
'TAT':'Y', 'TAC':'Y',
'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
'ATG':'>',
'TAA':'<', 'TGA':'<', 'TAG':'<'}
# Start '>', Stop '<'

def int_from_bytes(blob):
	size = len(blob)
	if size == 0:
		return 0
	return int.from_bytes(blob, "big", signed=True)

def convert_to_dna(x, base=8, gen=None):
	'''Takes and input 'x' and returns binary and genetic basepairs
	padded based on 'base' length, ie. base=8 will be in byte length.
	2 should be the minimum base
	'''
	if gen == None:
		gen = GEN
	else:
		gen = gen

	if type(x) == str:
		x = int_from_bytes(x.encode())
	if type(x) == bytes:
		x = int_from_bytes(x)
	if type(x) == float:
		raise ValueError 
	if type(x) == int:
		x = bin(x)

	y = f'{x[2:]}'
	q = len(y) /  base
	r = len(y) // base

	if q > r:
		y = y.zfill((int(q) + 1) * base)
	else:
		y = y.zfill(int(q) * base)
	
	out = []
	for i in range(len(y)):
		start = i * 2
		stop = (i+1) * 2
		if stop > len(y):
			break
		code = y[start:stop]
		bp = gen[code]
		out.append(bp)

	if len(y) % 2 != 0:
		raise ValueError(f'Error, an odd length integer, <{y}>, requires base value, <{base}>, to be even.')
	else:	
		return ''.join(out), y, x	

def convert_to_amino(x, code=None):
	# https://en.wikipedia.org/wiki/DNA_and_RNA_codon_tables
	if code == None:
		code = CODONS
	else:
		code = code

	protien = []
	for i in range(len(x)):
		start = i * 3
		stop = (i + 1) * 3
		if stop > len(x):
			break
		else:
			codon = x[start:stop]
		if codon in code.keys():
			amino = code[codon]
		else:
			amino = '_'
		protien.append(amino)
	
	return ''.join(protien)

	
#%%
# r = '15'
r = 'robert canar'

p = convert_to_dna(r, base=2)[0]
p
# %%
convert_to_amino(p, CODONS)



# %%
