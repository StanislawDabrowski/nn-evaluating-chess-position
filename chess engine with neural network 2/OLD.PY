'''
def read_positions(file_path):
	positions = []
	number_of_positions=0
	with open(file_path, "r") as file:
		for line in file:
			positions.append(line.strip())
	return positions

def read_evaluations(file_path):
	evaluations = []
	with open(file_path, "r") as file:
		for line in file:
			evaluations.append(line.strip())
	return evaluations

positions = read_positions("D:/private/chess engine/data/positions.fen")
print(f"number of positions: {len(positions)}")
print("succesfuly loaded positions")
evaluations = read_evaluations("D:/private/chess engine/data/evaluations.txt")
print("succesfuly loaded evaluations")

#x = torch.zeros(len(positions), 790, dtype=torch.bool)
x_temp = bitarray(790*4)



def get_piece_index(piece):
	if piece == "P":
		return 0
	if piece == "N":
		return 1
	if piece == "B":
		return 2
	if piece == "R":
		return 3
	if piece == "Q":
		return 4
	if piece == "K":
		return 5
	if piece == "p":
		return 6
	if piece == "n":
		return 7
	if piece == "b":
		return 8
	if piece == "r":
		return 9
	if piece == "q":
		return 10
	if piece == "k":
		return 11
	return -1


start_time = time.time()

start_time = time.time()
chunk_index = 0

with open("D:/private/chess engine/data/x.bin", "ab") as f:
	for i in range(len(positions)):
		if i%100000==0:
			print(f"converting positions {i/len(positions)*100}%   " + str(time.time() - start_time) + " seconds")
			start_time = time.time()
		board = positions[i].split(" ")[0]
		rank = 0
		file = 0
		for j in range(len(board)):
			if board[j].isdigit():
				file+=int(board[j])
			elif board[j] == "/":
				rank+=1
				file = 0
			else:
				index = get_piece_index(board[j])
				if index >= 0:
					x_temp[chunk_index*790+rank*8*12+file*12+index] = True
				file+=1
		move = positions[i].split(" ")[1]
		if move=="w":
			x_temp[chunk_index*790+8*8*12]=True
		else:
			x_temp[chunk_index*790+8*8*12+1]=True

		castle = positions[i].split(" ")[2]
		if "K" in castle:
			x_temp[chunk_index*790+8*8*12+2]=True
		if "Q" in castle:
			x_temp[chunk_index*790+8*8*12+3]=True
		if "k" in castle:
			x_temp[chunk_index*790+8*8*12+4]=True
		if "q" in castle:
			x_temp[chunk_index*790+8*8*12+5]=True

		en_passant = positions[i].split(" ")[3]
		if en_passant!="-":
			if en_passant[1]=="6":
				rank = 1
			else:
				rank = 0

			file = ord(en_passant[0])-97
			x_temp[chunk_index*790+8*8*12+6+rank*8+file]=True
	
		"""
		bin_representation = bin(int(positions[i].split(" ")[4]))[2:]
		while len(bin_representation)<8:
			bin_representation = "0"+bin_representation
		for j in range(8):
			if bin_representation[j]=="1":
				x_temp[8*8*12+6+16+j]=True
		"""
		if chunk_index==3:
			x_temp.tofile(f)
			x_temp.setall(0)
			chunk_index=0
		else:
			chunk_index+=1


print("saved x")
quit()
'''
"""
for i in range(len(positions)):
	board = positions[i].split(" ")[0]
	rank = 0
	file = 0
	for j in range(len(board)):
		if board[j].isdigit():
			file+=int(board[j])
		elif board[j] == "/":
			rank+=1
			file = 0
		else:
			index = get_piece_index(board[j])
			if index >= 0:
				x[i][rank*8*12+file*12+index] = True
			file+=1
	move = positions[i].split(" ")[1]
	if move=="w":
		x[i][8*8*12]=True
	else:
		x[i][8*8*12+1]=True

	castle = positions[i].split(" ")[2]
	if "K" in castle:
		x[i][8*8*12+2]=True
	if "Q" in castle:
		x[i][8*8*12+3]=True
	if "k" in castle:
		x[i][8*8*12+4]=True
	if "q" in castle:
		x[i][8*8*12+5]=True

	en_passant = positions[i].split(" ")[3]
	if en_passant!="-":
		if en_passant[1]=="6":
			rank = 1
		else:
			rank = 0

		file = ord(en_passant[0])-97
		x[i][8*8*12+6+rank*8+file]=True
	bin_representation = bin(int(positions[i].split(" ")[4]))[2:]
	while len(bin_representation)<8:
		bin_representation = "0"+bin_representation
	for j in range(8):
		if bin_representation[j]=="1":
			x[i][8*8*12+6+16+j]=True
"""
"""
print("convertion cpu: " + str(time.time() - start_time) + " seconds")

#save tensor at D:/private/chess engine/positions.pt
torch.save(x, "D:/private/chess engine/positions.pt")

y = torch.tensor([[float(evaluation)] for evaluation in evaluations])

#save tensor at D:/private/chess engine/evaluations.pt
torch.save(y, "D:/private/chess engine/evaluations.pt")
"""




"""
	class Net_v1(nn.Module):
		def __init__(self):
			super(Net_v1, self).__init__()
			#self.max_value = 10000
			self.s = nn.Sequential(
				nn.Linear(790, 2048),
				nn.LeakyReLU(),
				nn.Linear(2048, 2048),
				nn.LeakyReLU(),
				nn.Linear(2048, 2048),
				nn.LeakyReLU(),
				nn.Linear(2048, 256),
				nn.LeakyReLU(),
				nn.Linear(256, 32),
				nn.LeakyReLU(),
				nn.Linear(32, 1),
				nn.Tanh()
			)
		def forward(self, x):
			x = x.view(-1, 790)
			x = self.s(x)
			return x
	

	class Net_v2(nn.Module):
		def __init__(self):
			super(Net_v2, self).__init__()
			#self.max_value = 10000
			self.s = nn.Sequential(
				nn.Linear(790, 4096),
				nn.LeakyReLU(),
				nn.Linear(4096, 512),
				nn.LeakyReLU(),
				nn.Linear(512, 128),
				nn.LeakyReLU(),
				nn.Linear(128, 32),
				nn.LeakyReLU(),
				nn.Linear(32, 1),
				nn.Tanh()
			)
		def forward(self, x):
			x = x.view(-1, 790)
			x = self.s(x)
			#x = x*self.max_value
			return x
	

	class Net_v3(nn.Module):
		def __init__(self):
			super(Net_v3, self).__init__()
			#self.max_value = 10000
			self.s = nn.Sequential(
				nn.Linear(790, 4096),
				nn.ReLU(),
				nn.Linear(4096, 512),
				nn.ReLU(),
				nn.Linear(512, 128),
				nn.ReLU(),
				nn.Linear(128, 32),
				nn.ReLU(),
				nn.Linear(32, 1),
				nn.Tanh()
			)
		def forward(self, x):
			x = x.view(-1, 790)
			x = self.s(x)
			#x = x*self.max_value
			return x

	class Net_v4(nn.Module):#sparse layers
		def __init__(self, mask1, mask2):
			super(Net_v4, self).__init__()
			self.s = nn.Sequential(
				SparseConnectedLayer(790, 8192, mask1, 'leaky_relu', 0.01),
				nn.LeakyReLU(),
				SparseConnectedLayer(8192, 8192, mask2, 'leaky_relu', 0.01),
				nn.LeakyReLU(),
				nn.Linear(8192, 2024),
				nn.LeakyReLU(),
				nn.Linear(2024, 512),
				nn.LeakyReLU(),
				nn.Linear(512, 64),
				nn.LeakyReLU(),
				nn.Linear(64, 1),
				nn.Tanh()
			)
		def forward(self, x):
			x = x.view(-1, 790)
			x = self.s(x)
			return x
	"""
'''
class Net_v5(nn.Module):#sparse layers and embeddings
	def __init__(self, mask1, mask2):
		super(Net_v5, self).__init__()
		self.embedding = nn.Embedding(13, 8)
		self.s = nn.Sequential(
			SparseConnectedLayer(534, 4096, mask1, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			SparseConnectedLayer(4096, 4096, mask2, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(4096, 1024),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(1024, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
	def forward(self, x1, x2):
		x1 = self.embedding(x1)#size (batch_size, 64, 8))
		x1 = x1.view(x1.size(0), -1)#size (batch_size, 512)
		x = torch.cat((x1, x2), 1)#size (batch_size, 534)
		return self.s(x)
'''
'''
class Net_v6(nn.Module):#sparse layers and embeddings
	def __init__(self, mask1, mask2):
		super(Net_v6, self).__init__()
		self.embedding = nn.Embedding(13, 6)
		self.s = nn.Sequential(
			SparseConnectedLayer(406, 2048, mask1, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			SparseConnectedLayer(2048, 2048, mask2, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(2048, 512),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(512, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
	def forward(self, x1, x2):
		x1 = self.embedding(x1)#size (batch_size, 64, 8))
		x1 = x1.view(x1.size(0), -1)#size (batch_size, 512)
		x = torch.cat((x1, x2), 1)#size (batch_size, 534)
		return self.s(x)
'''
'''
class Net_v7(nn.Module):#net_v5 with batch normalization
	def __init__(self, mask1, mask2):
		super(Net_v7, self).__init__()
		self.embedding = nn.Embedding(13, 8)
		self.s = nn.Sequential(
			SparseConnectedLayer(534, 4096, mask1, 'leaky_relu', 0.01),
			nn.BatchNorm1d(4096),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			SparseConnectedLayer(4096, 4096, mask2, 'leaky_relu', 0.01),
			nn.BatchNorm1d(4096),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(4096, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(),
			nn.Dropout(p=0.4),
			nn.Linear(1024, 256),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(),
			nn.Linear(256, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
	def forward(self, x1, x2):
		x1 = self.embedding(x1)#size (batch_size, 64, 8))
		x1 = x1.view(x1.size(0), -1)#size (batch_size, 512)
		x = torch.cat((x1, x2), 1)#size (batch_size, 534)
		return self.s(x)
'''

'''
class Net_v8(nn.Module):
	def __init__(self, mask1, mask2):
		super(Net_v8, self).__init__()
		self.embedding = nn.Embedding(13, 8)
		self.s = nn.Sequential(
			SparseConnectedLayer(534, 6144, mask1, 'leaky_relu', 0.01),
			nn.BatchNorm1d(6144),
			nn.LeakyReLU(),
			nn.Dropout(p=0.5),
			SparseConnectedLayer(6144, 6144, mask2, 'leaky_relu', 0.01),
			nn.BatchNorm1d(6144),
			nn.LeakyReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(6144, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 256),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(),
			nn.Linear(256, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
	def forward(self, x1, x2):
		x1 = self.embedding(x1)#size (batch_size, 64, 8))
		x1 = x1.view(x1.size(0), -1)#size (batch_size, 512)
		x = torch.cat((x1, x2), 1)#size (batch_size, 534)
		return self.s(x)
'''
'''
class Net_v9(nn.Module):
	def __init__(self, mask1, mask2):
		super(Net_v9, self).__init__()
		self.embedding = nn.Embedding(13, 8)
		self.s = nn.Sequential(
			SparseConnectedLayer(534, 16384, mask1, 'leaky_relu', 0.01),
			nn.BatchNorm1d(16384),
			nn.LeakyReLU(),
			nn.Dropout(p=0.5),
			SparseConnectedLayer(16384, 16384, mask2, 'leaky_relu', 0.01),
			nn.BatchNorm1d(16384),
			nn.LeakyReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(16384, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 256),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(),
			nn.Linear(256, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
	def forward(self, x1, x2):
		x1 = self.embedding(x1)#size (batch_size, 64, 8))
		x1 = x1.view(x1.size(0), -1)#size (batch_size, 512)
		x = torch.cat((x1, x2), 1)#size (batch_size, 534)
		return self.s(x)
'''
'''
class Net_v10(nn.Module):#net_v5 with batch normalization
	def __init__(self, mask1, mask2):
		super(Net_v10, self).__init__()
		self.embedding = nn.Embedding(13, 8)
		self.s = nn.Sequential(
			SparseConnectedLayer(534, 16384, mask1, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			SparseConnectedLayer(16384, 16384, mask2, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			nn.Linear(16384, 1024),
			nn.LeakyReLU(),
			nn.Linear(1024, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
	def forward(self, x1, x2):
		x1 = self.embedding(x1)#size (batch_size, 64, 8))
		x1 = x1.view(x1.size(0), -1)#size (batch_size, 512)
		x = torch.cat((x1, x2), 1)#size (batch_size, 534)
		return self.s(x)
'''

'''
def calculate_mask1():
	mask = torch.zeros(65, 790)
	for i in range(64):
		for j in range(6):
			mask[i][64*12+j] = 1#turn and castling rights

	for i in range(64):
		x = i%8
		y = i//8
		mask[i][y*8*12:(y+1)*8*12:] = 1
		for j in range(8):
			mask[i][j*12*8+x*12:j*12*8+x*12+12] = 1
		for j in range(8):
			for k in range(8):
				if j+k==x+y or j-k==x-y:
					mask[i][j*12+k*12*8:(j*12+k*12*8)+12] = 1
		if x>1:
			if y>1:
				mask[i][(y-2)*8*12+(x-1)*12:(y-2)*8*12+(x-1)*12+12] = 1
				mask[i][(y-1)*8*12+(x-2)*12:(y-1)*8*12+(x-2)*12+12] = 1
			elif y>0:
				mask[i][(y-1)*8*12+(x-2)*12:(y-1)*8*12+(x-2)*12+12] = 1
			if y<6:
				mask[i][(y+2)*8*12+(x-1)*12:(y+2)*8*12+(x-1)*12+12] = 1
				mask[i][(y+1)*8*12+(x-2)*12:(y+1)*8*12+(x-2)*12+12] = 1
			elif y<7:
				mask[i][(y+1)*8*12+(x-2)*12:(y+1)*8*12+(x-2)*12+12] = 1
		elif x>0:
			if y>1:
				mask[i][(y-2)*8*12+(x-1)*12:(y-2)*8*12+(x-1)*12+12] = 1
			if y<6:
				mask[i][(y+2)*8*12+(x-1)*12:(y+2)*8*12+(x-1)*12+12] = 1
		if x<6:
			if y>1:
				mask[i][(y-2)*8*12+(x+1)*12:(y-2)*8*12+(x+1)*12+12] = 1
				mask[i][(y-1)*8*12+(x+2)*12:(y-1)*8*12+(x+2)*12+12] = 1
			elif y>0:
				mask[i][(y-1)*8*12+(x+2)*12:(y-1)*8*12+(x+2)*12+12] = 1
			if y<6:
				mask[i][(y+2)*8*12+(x+1)*12:(y+2)*8*12+(x+1)*12+12] = 1
				mask[i][(y+1)*8*12+(x+2)*12:(y+1)*8*12+(x+2)*12+12] = 1
			elif y<7:
				mask[i][(y+1)*8*12+(x+2)*12:(y+1)*8*12+(x+2)*12+12] = 1
		elif x<7:
			if y>1:
				mask[i][(y-2)*8*12+(x+1)*12:(y-2)*8*12+(x+1)*12+12] = 1
			if y<6:
				mask[i][(y+2)*8*12+(x+1)*12:(y+2)*8*12+(x+1)*12+12] = 1
	mask[64][64*12:790] = 1
	return mask

"""
def calculate_mask2():
	mask = torch.zeros(4096, 8192)
	for i in range(64):
		for j in range(63):
			mask[i*63+j][i*127:(i+1)*127] = 1
	for i in range(64):
		mask[64*63+i][64*127:64*127+64] = 1
	return mask
"""

def calculate_mask2():
	mask = torch.zeros(65, 8192)
	for i in range(64):
		for j in range(64):
			mask[i][64*127+j] = 1#turn and castling rights

	for i in range(64):
		x = i%8
		y = i//8
		mask[i][y*8*127:(y+1)*8*127:] = 1
		for j in range(8):
			mask[i][j*127*8+x*127:j*127*8+x*127+127] = 1
		for j in range(8):
			for k in range(8):
				if j+k==x+y or j-k==x-y:
					mask[i][j*127+k*127*8:(j*127+k*127*8)+127] = 1
		if x>1:
			if y>1:
				mask[i][(y-2)*8*127+(x-1)*127:(y-2)*8*127+(x-1)*127+127] = 1
				mask[i][(y-1)*8*127+(x-2)*127:(y-1)*8*127+(x-2)*127+127] = 1
			elif y>0:
				mask[i][(y-1)*8*127+(x-2)*127:(y-1)*8*127+(x-2)*127+127] = 1
			if y<6:
				mask[i][(y+2)*8*127+(x-1)*127:(y+2)*8*127+(x-1)*127+127] = 1
				mask[i][(y+1)*8*127+(x-2)*127:(y+1)*8*127+(x-2)*127+127] = 1
			elif y<7:
				mask[i][(y+1)*8*127+(x-2)*127:(y+1)*8*127+(x-2)*127+127] = 1
		elif x>0:
			if y>1:
				mask[i][(y-2)*8*127+(x-1)*127:(y-2)*8*127+(x-1)*127+127] = 1
			if y<6:
				mask[i][(y+2)*8*127+(x-1)*127:(y+2)*8*127+(x-1)*127+127] = 1
		if x<6:
			if y>1:
				mask[i][(y-2)*8*127+(x+1)*127:(y-2)*8*127+(x+1)*127+127] = 1
				mask[i][(y-1)*8*127+(x+2)*127:(y-1)*8*127+(x+2)*127+127] = 1
			elif y>0:
				mask[i][(y-1)*8*127+(x+2)*127:(y-1)*8*127+(x+2)*127+127] = 1
			if y<6:
				mask[i][(y+2)*8*127+(x+1)*127:(y+2)*8*127+(x+1)*127+127] = 1
				mask[i][(y+1)*8*127+(x+2)*127:(y+1)*8*127+(x+2)*127+127] = 1
			elif y<7:
				mask[i][(y+1)*8*127+(x+2)*127:(y+1)*8*127+(x+2)*127+127] = 1
		elif x<7:
			if y>1:
				mask[i][(y-2)*8*127+(x+1)*127:(y-2)*8*127+(x+1)*127+127] = 1
			if y<6:
				mask[i][(y+2)*8*127+(x+1)*127:(y+2)*8*127+(x+1)*127+127] = 1
	mask[64][::] = 1
	return mask

def print_board_from_mask(mask):
	for i in range(64):
		for j in range(12):
			if mask[i*12+j]==1:
				print("1", end="")
				break
			elif j==11:
				print("0", end="")
		if i%8==7:
			print()
'''

"""
def smooth_max(x, y, beta=20):
	return (1 / beta) * torch.log(torch.exp(beta * x) + torch.exp(beta * y))

def differentiable_abs(a, epsilon=0.000001):
	return torch.sqrt(a ** 2 + epsilon)
"""

"""
	class squareRootLoss(nn.Module):
		def __init__(self, epsilon=0.000001):
			super(squareRootLoss, self).__init__()
			self.epsilon = epsilon

		def forward(self, predictions, targets):
			loss = torch.sqrt(torch.sqrt((predictions - targets) ** 2+self.epsilon))
			return loss.mean()  # Return the mean loss across the batch
	"""
"""
	class FourthRootLossSmoothMax(nn.Module):
		def __init__(self, alpha=0.01, epsilon=0.000001):
			super(FourthRootLossSmoothMax, self).__init__()
			self.epsilon = epsilon
			self.substract_value = torch.sqrt(torch.sqrt(smooth_max(torch.sqrt(epsilon), alpha)))#substracted to make the loss 0 when the prediction is equal to the target

		def forward(self, predictions, targets):
			loss = torch.sqrt(torch.sqrt(smooth_max(torch.sqrt((predictions - targets) ** 2+self.epsilon), torch.tensor([0.01]))))-self.substract_value
			return loss.mean()  # Return the mean loss across the batch
	"""


'''
	class CustomDataset(Dataset):
	
	def __init__(self, x_path, y_path, x_bits_size, y_bits_size, samples_number):
		self.x_path = x_path
		self.y_path = y_path
		self.x_bits_size = x_bits_size
		self.y_bits_size = y_bits_size
		self.samples_number = samples_number
		
		
		with open(x_path, "rb") as f:
			self.x_bytes = f.read()
		with open(y_path, "rb") as f:
			self.y_bytes = f.read()
		
		
		"""
		self.x = torch.zeros(((len(x_bytes)*8)//x_bits_size, x_bits_size), dtype=torch.bool)
		index = 0
		for i in range(len(x_bytes)):
			for j in range(8):
				self.x[index // x_bits_size][index % x_bits_size] = bool(x_bytes[i] & (1 << (7-j)))
				index+=1
		self.y = torch.zeros(((len(y_bytes)*8)//y_bits_size, y_bits_size), dtype=torch.bool)
		self.y = torch.frombuffer(y_bytes, dtype=torch.int16)
		"""


	def __len__(self):
		return self.samples_number

	def __getitem__(self, idx):
		x = torch.zeros((self.x_bits_size, 1), dtype=torch.float)
		index = -((idx*self.x_bits_size)%8)
		
		with open(self.x_path, "rb") as f:
			f.seek((idx*self.x_bits_size)//8)
			x_bytes = f.read(self.x_bits_size//8+8)
		for i in range(len(x_bytes)):
			for j in range(8):
				if index>0:
					x[i] = bool(x_bytes[i] & (1 << (7-j)))
					if index>self.x_bits_size-1:
						break
				index+=1

		#ASSUMPTION THAT Y IS 16 BITS!!!
		y = torch.zeros(1, dtype=torch.float)
		with open(self.y_path, "rb") as f:
			f.seek(idx*2)
			y_bytes = f.read(2)
		int_value = int.from_bytes(y_bytes, byteorder='little', signed=True)
		y[0] = int_value
		return x, y
	'''
"""
	def __init__(self, x_path, y_path, x_bits_size, y_bits_size, samples_number):
		self.x_bits_size = x_bits_size
		self.y_bits_size = y_bits_size
		self.samples_number = samples_number
		
		
		with open(x_path, "rb") as f:
			self.x_bytes = f.read()
		with open(y_path, "rb") as f:
			self.y_bytes = f.read()


	def __len__(self):
		return self.samples_number

	def __getitem__(self, idx):
		x = torch.zeros((self.x_bits_size, 1), dtype=torch.float)
		index = idx*self.x_bits_size-((idx*self.x_bits_size)%8)


		x_bytes_temp = self.x_bytes[(idx*self.x_bits_size)//8:(idx*self.x_bits_size+self.x_bits_size)//8+8:]
		for i in range(len(x_bytes_temp)):
			for j in range(8):
				if index>idx*self.x_bits_size-1:
					x[i] = bool(x_bytes_temp[i] & (1 << (7-j)))
					if index>idx*self.x_bits_size+self.x_bits_size-1:
						break
				index+=1

		#ASSUMPTION THAT Y IS 16 BITS!!!
		y = torch.zeros(1, dtype=torch.float)
		int_value = int.from_bytes(self.y_bytes[idx*2:idx*2+2:], byteorder='little', signed=True)
		y[0] = int_value
		return x, y
	"""

"""
	def __init__(self, x_path, y_path, x_bits_size, y_bits_size, samples_number):
		self.x_bits_size = x_bits_size
		self.y_bits_size = y_bits_size
		self.samples_number = samples_number
		
		self.x_bits = bitarray()
		with open(x_path, "rb") as f:
			self.x_bits.fromfile(f)
		with open(y_path, "rb") as f:
			self.y_bytes = f.read()

			
		#self.x_bits = np.unpackbits(np.frombuffer(x_bytes, dtype=np.uint8))


	def __len__(self):
		return self.samples_number

	def __getitem__(self, idx):
		# Get the x bits corresponding to the sample
		start_idx = idx * self.x_bits_size
		end_idx = start_idx + self.x_bits_size
		x_bits = self.x_bits[start_idx:end_idx]
		x = torch.tensor(x_bits, dtype=torch.float32).view(self.x_bits_size, 1)

		# Assuming y is 16 bits
		y_bytes = self.y_bytes[idx * 2 : idx * 2 + 2]
		y_value = int.from_bytes(y_bytes, byteorder='little', signed=True)
		y = torch.tensor([y_value/10000], dtype=torch.float32)

		return x, y
	"""



"""
#compressed_mask1 = calculate_mask1(8)
	#mask1 = torch.cat((compressed_mask1.unsqueeze(1).repeat(1, 63, 1).view(-1, compressed_mask1.size(1)), compressed_mask1[64].unsqueeze(0)), dim=0)
	
	#compressed_mask1 = calculate_mask1(6)
	#mask1 = torch.cat((compressed_mask1.unsqueeze(1).repeat(1, 31, 1).view(-1, compressed_mask1.size(1)), compressed_mask1[64].unsqueeze(0).repeat(33, 1)), dim=0)
	
	
	#compressed_mask1 = calculate_mask1(8)
	#mask1 = torch.cat((compressed_mask1.unsqueeze(1).repeat(1, 94, 1).view(-1, compressed_mask1.size(1)), compressed_mask1[64].unsqueeze(0).repeat(34, 1)), dim=0)
	
	
	#mask2 = calculate_mask2()
	#compressed_mask2 = calculate_mask2()
	#mask2 = torch.cat((compressed_mask2.unsqueeze(1).repeat(1, 63, 1).view(-1, compressed_mask2.size(1)) ,compressed_mask2[64].unsqueeze(0)), dim=0)
	
	#compressed_mask2 = calculate_mask2(2048)
	#mask2 = torch.cat((compressed_mask2.unsqueeze(1).repeat(1, 31, 1).view(-1, compressed_mask2.size(1)) ,compressed_mask2[64].unsqueeze(0).repeat(33, 1)), dim=0)

	
"""

"""
for epoch in range(epochs):
	print(epoch)
	if epoch%testing_iterations==0:
		with torch.inference_mode():
			net.eval()
			running_loss_test = 0.0
			for test_batch in test_dataloader:
				output = net(test_batch[0].to(device))
				loss = criterion(output, test_batch[1].to(device))
				running_loss_test += loss.item()
			test_losses.append(running_loss_test/(len(test_dataset)/batch_size))
			print(f"test loss: {running_loss_test/(len(test_dataset)/batch_size)}")
			#train loss
			print(f"train loss: {running_loss/(len(dataset)/batch_size)}")
			running_loss = 0.0
			net.train()
	batches_preceded = 0
	for batch in dataloader:
		if batches_preceded%((train_elements_number/batch_size)//1000)==0:
			print(f"batches preceded: {batches_preceded} {(batches_preceded/((train_elements_number/batch_size)//100)):.2f}% {(time.time()-start_time):.2f} seconds")
			start_time = time.time()
		optimizer.zero_grad()
		output = net(batch[0].to(device))
		loss = criterion(output, batch[1].to(device))
		running_loss += loss.item()
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())
		batches_preceded+=1
	if epoch%saving_iterations==0:
		iterations+=saving_iterations
		save_model(saving_path+"/"+str(iterations)+".pth", epoch, net.state_dict, optimizer.state_dict, scheduler.state_dict, test_losses[-1])
		with open(saving_path+"/iterations.txt", "w") as f:
			f.write(str(iterations))
"""