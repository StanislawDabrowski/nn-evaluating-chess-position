import os
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from datetime import datetime
from bitarray import bitarray
from torch.utils.data import Dataset, DataLoader, dataset
import numpy as np

class CustomDataset(Dataset):
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


def main():
	
	if torch.cuda.is_available():
		print("\033[32mCUDA is available\033[0m\n\n")
		device = "cuda"
	else:
		device = "cpu"
		print("\033[31mCUDA is unavailable!\033[0m\n\n")


	input_bin_file_test = 'D:/private/chess engine/data/x_test.bin'
	label_bin_file_test = 'D:/private/chess engine/data/y_test.bin'

	input_size = 790
	label_size = 16
	test_elements_number = 1660516

	#test dataset
	test_dataset = CustomDataset(input_bin_file_test, label_bin_file_test, input_size, label_size, test_elements_number)


	batch_size = int(input())
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
			super(Net, self).__init__()
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

	"""
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
		def __init__(self, in_features, out_features, mask):
			super(SparseConnectedLayer, self).__init__()
			self.weight = nn.Parameter(torch.randn(out_features, in_features))
			self.bias = nn.Parameter(torch.randn(out_features))
			self.register_buffer('mask', mask)
    
		def forward(self, x):
			masked_weight = self.weight * self.mask
			return torch.matmul(x, masked_weight.T) + self.bias

	
	class Net_v4(nn.Module):#sparse layers
		def __init__(self, mask1, mask2):
			super(Net_v4, self).__init__()
			self.s = nn.Sequential(
				SparseConnectedLayer(790, 8192, mask1),
				nn.LeakyReLU(),
				SparseConnectedLayer(8192, 4096, mask2),
				nn.LeakyReLU(),
				nn.Linear(4096, 2024),
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

	def calculate_mask2():
		mask = torch.zeros(4096, 8192)
		for i in range(64):
			for j in range(63):
				mask[i*63+j][i*127:(i+1)*127] = 1
		for i in range(64):
			mask[64*63+i][64*127:64*127+64] = 1
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



	compressed_mask = calculate_mask1()
	mask1 = compressed_mask.unsqueeze(1).repeat(1, 127, 1).view(-1, compressed_mask.size(1))[:-63:]
	mask2 = calculate_mask2()
	"""

	class squareRootLoss(nn.Module):
		def __init__(self, epsilon=0.000001):
			super(squareRootLoss, self).__init__()
			self.epsilon = epsilon

		def forward(self, predictions, targets):
			loss = torch.sqrt(torch.sqrt((predictions - targets) ** 2+self.epsilon))
			return loss.mean()  # Return the mean loss across the batch

	
	#net = Net_v4(mask1, mask2)
	net = Net_v3()
	saving_path = ""
	saving_path = "D:/private/chess engine/nets/best_for_now"
	if saving_path=="":
		starting_program_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		saving_path = "D:/private/chess engine/nets/"+str(starting_program_time)

		os.makedirs(saving_path, exist_ok=True)

		iterations = 0
		with open(saving_path+"/iterations.txt", "w") as f:
			f.write(str(iterations))
	else:
		net.load_state_dict(torch.load(saving_path+"/9.pth")['model_state_dict'])
		#print loss
		print(torch.load(saving_path+"/9.pth")['test_loss'])


	
	net = net.to(device)

	#criterion = squareRootLoss()
	criterion = nn.MSELoss()


	
	abs_error_sum = 0
	relative_abs_error_sum = 0
	
	start_time = time.time()
	iterations=0



	with torch.inference_mode():
		net.eval()
		running_loss_test = 0.0
		for test_batch in test_dataloader:
			#batch_label = test_batch[1].to(device)
			output = net(test_batch[0].to(device))

			
			loss = criterion(output, test_batch[1].to(device))
			running_loss_test += loss.item()
			# Move test_batch[1] to device once
			y_true = test_batch[1].to(device)
			y_pred = output

			# Apply condition: Only consider y_true values between -1 and 1
			condition = (y_true > -1) & (y_true <1)

			# Apply the condition to filter both predictions and true values
			filtered_y_true = y_true[condition]
			filtered_y_pred = y_pred[condition]

			# Compute absolute error sum and relative absolute error sum only for filtered values
			if len(filtered_y_true) > 0:  # Ensure there are valid elements after filtering
				abs_error_sum += torch.sum(torch.abs(filtered_y_pred - filtered_y_true)).item()
				relative_abs_error_sum += torch.sum(torch.abs(filtered_y_pred - filtered_y_true) /
													torch.maximum(torch.abs(filtered_y_true), torch.tensor(0.0100, device=device))).item()

			#abs_error_sum += torch.sum(torch.abs(output - test_batch[1].to(device))).item()
			#relative_abs_error_sum += torch.sum(torch.abs(output - test_batch[1].to(device)) / torch.maximum(torch.abs(test_batch[1].to(device)), torch.tensor(0.0100, device=device))).item()
			#for i in range(len(test_batch[1])):
				#print(f"{(output[i].item()*10000):.0f} {(test_batch[1][i].item()*10000):.0f}")
			print("|", end="")
		
	print(f"test loss: {running_loss_test/(len(test_dataset)/batch_size)}")
	print(f"abs error: {abs_error_sum/(len(test_dataset))}")
	print(f"relative abs error: {relative_abs_error_sum/(len(test_dataset))}")
	input()


if __name__ == "__main__":
	main()