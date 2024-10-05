import os
import torch
import torch.nn as nn
import time
#import matplotlib.pyplot as plt
from datetime import datetime
from bitarray import bitarray
from torch.utils.data import Dataset, DataLoader, dataset
#import numpy as np







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
random_t = torch.rand(798, 1)

start_time = time.time()

for i in range(100):
	torch.mm(x, random_t)

print("cpu: " + str(time.time() - start_time) + " seconds")#approximately 15 seconds

x = x.to("cuda")
random_t = random_t.to("cuda")


start_time = time.time()

for i in range(100):
	torch.mm(x, random_t)

print("cuda: " + str(time.time() - start_time) + " seconds")#approximately 0.03 seconds
"""





class CustomDataset(Dataset):
	'''
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


def save_model(path, epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, test_loss):
	torch.save({
    'epoch': epoch,
    'model_state_dict': model_state_dict(),
    'optimizer_state_dict': optimizer_state_dict(),
    'scheduler_state_dict': scheduler_state_dict(),
	'test_loss': test_loss,
}, path)


def main():
	
	if torch.cuda.is_available():
		print("\033[32mCUDA is available\033[0m\n\n")
		device = "cuda"
	else:
		device = "cpu"
		print("\033[31mCUDA is unavailable!\033[0m\n\n")

	

	input_bin_file_train = 'D:/private/chess engine/data/x_train.bin'
	label_bin_file_train = 'D:/private/chess engine/data/y_train.bin'
	input_bin_file_test = 'D:/private/chess engine/data/x_test.bin'
	label_bin_file_test = 'D:/private/chess engine/data/y_test.bin'

	input_size = 790
	label_size = 16
	train_elements_number = 81365220
	test_elements_number = 1660516

	#train dataset
	#dataset = CustomDataset(input_bin_file_train, label_bin_file_train, input_size, label_size, train_elements_number)
	#test dataset
	test_dataset = CustomDataset(input_bin_file_test, label_bin_file_test, input_size, label_size, test_elements_number)


	batch_size = int(input())
	#train dataloader
	dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
	#test dataloader
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


	print("succesfuly loaded data")
	#time.sleep(2)
	os.system("cls")



	class squareRootLoss(nn.Module):
		def __init__(self, epsilon=0.000001):
			super(squareRootLoss, self).__init__()
			self.epsilon = epsilon

		def forward(self, predictions, targets):
			loss = torch.sqrt(torch.sqrt((predictions - targets) ** 2+self.epsilon))
			return loss.mean()  # Return the mean loss across the batch

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
	"""
	
	class SparseConnectedLayer(nn.Module):
		def __init__(self, in_features, out_features, mask, nonlinearity, negative_slope=0):
			super(SparseConnectedLayer, self).__init__()
			#initialize weights using torch.nn.init.kaiming_normal_()
			self.weight = nn.Parameter(torch.zeros(out_features, in_features))
			torch.nn.init.kaiming_normal_(self.weight, a=negative_slope, nonlinearity=nonlinearity)
			self.bias = nn.Parameter(torch.zeros(out_features))
			self.register_buffer('mask', mask)
    
		def forward(self, x):
			masked_weight = self.weight * self.mask
			return torch.sparse.mm(x, masked_weight.T.to_sparse()) + self.bias

	
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



	compressed_mask1 = calculate_mask1()
	mask1 = compressed_mask1.unsqueeze(1).repeat(1, 127, 1).view(-1, compressed_mask1.size(1))[:-63:]
	#mask2 = calculate_mask2()
	compressed_mask2 = calculate_mask2()
	mask2 = compressed_mask2.unsqueeze(1).repeat(1, 127, 1).view(-1, compressed_mask2.size(1))[:-63:]
	
	print(mask1.size())
	#print(mask2.size())
	
	"""
	for i in range(64):
		print_board_from_mask(compressed_mask[i])
		print("\n")
	
	input()
	"""
			




	net = Net_v4(mask1, mask2)
	saving_path = ""
	#saving_path = "D:/private/chess engine/nets/2024-10-01_23-04-00/"
	if saving_path=="":
		starting_program_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		saving_path = "D:/private/chess engine/nets/"+str(starting_program_time)

		os.makedirs(saving_path, exist_ok=True)

		iterations = 0
		with open(saving_path+"/iterations.txt", "w") as f:
			f.write(str(iterations))
	else:
		net.load_state_dict(torch.load(saving_path+"11.pth")['model_state_dict'])


	
	net = net.to(device)

	
	epochs = 10

	#mean absolute errors loss
	criterion = squareRootLoss()
	#criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

	#for param_group in optimizer.param_groups:
	#	param_group['lr'] = 0.0015

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	train_losses = []
	test_losses = []

	testing_iterations = 1
	saving_iterations = 1
	

	
	start_time = time.time()
	running_loss = 0.0
	iterations=0
	#save hyperparamyters to hyperparameters.txt
	with open(saving_path+"/hyperparameters.txt", "w") as f:
		f.write(f"batch_size: {batch_size}\n")
		f.write(f"optimizer: {optimizer}\n")
		f.write(f"criterion: {criterion}\n")
		f.write(f"device: {device}\n")
		f.write(f"train_elements_number: {train_elements_number}\n")
		f.write(f"test_elements_number: {test_elements_number}\n")
		f.write(f"input_bin_file_train: {input_bin_file_train}\n")
		f.write(f"label_bin_file_train: {label_bin_file_train}\n")
		f.write(f"input_bin_file_test: {input_bin_file_test}\n")
		f.write(f"label_bin_file_test: {label_bin_file_test}\n")
		f.write(f"input_size: {input_size}\n")
		f.write(f"label_size: {label_size}\n")
		f.write(f"saving_path: {saving_path}\n")
		f.write(f"scheduler_type: {type(scheduler).__name__}\n")

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
	if os.path.isfile(saving_folder_path + "/iterations.txt"):
		iterations = int(open(saving_folder_path+"/iterations.txt", "r").read())
	else:
		#create iterations.txt file with "0" in it
		with open(saving_folder_path+"/iterations.txt", "w") as file:
			file.write("0")
	"""

	"""
	batch_size = 1<<10
	for epoch in range(epochs):
		running_loss = 0.0
		for i in range(0, len(x_train)-batch_size, batch_size):
			optimizer.zero_grad()
			output = net(x_train[i:i+batch_size])
			loss = criterion(output, y_train[i:i+batch_size])
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		train_losses.append(running_loss/(len(x_train)//batch_size))
		if epoch%testing_iterations==0:
			print(f"epoch: {epoch}, loss: {running_loss/(len(x_train)//batch_size)}")
			running_loss = 0.0
			running_loss_test = 0.0
			for i in range(0, len(x_test), batch_size):
				output = net(x_test[i:i+batch_size])
				loss = criterion(output, y_test[i:i+batch_size])
				running_loss_test += loss.item()
			test_losses.append(running_loss_test/(len(x_test)//batch_size))
			print(f"test loss: {running_loss_test/(len(x_test)//batch_size)}")
		if epoch%saving_iterations==0:
			if epoch!=0:
				iterations+=saving_iterations
			torch.save(net.state_dict(), saving_path+"/"+str(iterations)+".pth")
			with open(saving_path+"/iterations.txt", "w") as f:
				f.write(str(iterations))
	"""


	"""
	#vusialize the loss on a plot
	plt.plot(train_losses, label="train")
	plt.plot(test_losses, label="test")
	plt.legend()
	plt.show()
	"""

if __name__ == "__main__":
	main()