import os
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from datetime import datetime
from bitarray import bitarray
from torch.utils.data import Dataset, DataLoader, dataset
import numpy as np

"""
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
"""

class CustomDataset(Dataset):
	def __init__(self, x_path, y_path, samples_number, x_bytes_values):
		self.x_path = x_path
		self.x_bytes_values = x_bytes_values
		self.data_in_vram = False
		self.samples_number = samples_number
		self.first_getitem_flag = True
		with open(y_path, "rb") as f:
			y_bytes = f.read()
		self.y = torch.frombuffer(y_bytes, dtype=torch.int16).view(samples_number, 1)
		y_bytes = None


	def __len__(self):
		return self.samples_number


	def __getitem__(self, idx):
		if self.first_getitem_flag:
			self.first_getitem_flag = False
			if self.data_in_vram:
				with open(self.x_path, "rb") as f:
					x_bytes = f.read()
				byte_array_np = np.frombuffer(x_bytes, dtype=np.uint8)

				# Reshape the array to (n, 86)
				reshaped_array = byte_array_np.reshape(self.samples_number, self.x_bytes_values)

				# Split the reshaped array into two tensors
				self.x1 = torch.tensor(reshaped_array[:, :64], dtype=torch.uint8).to('cuda')  # First 64 bytes
				self.x2 = torch.tensor(reshaped_array[:, 64:], dtype=torch.bool).to('cuda')    # Last 22 bytes

				x_bytes = None
				self.y = self.y.to('cuda')
			else:
				with open(self.x_path, "rb") as f:
					x_bytes = f.read()
				self.x = torch.frombuffer(x_bytes, dtype=torch.uint8).view(self.samples_number, self.x_bytes_values)
		"""
				if self.data_in_vram:
			return self.x1[idx].to(torch.int), self.x2[idx].to(torch.float), self.y[idx].to(torch.float)/200
		else:
			return self.x[idx][:64:].to("cuda").to(torch.int), self.x[idx][64::].to("cuda").to(torch.float), self.y[idx].to("cuda").to(torch.float)/200
		"""
		if self.data_in_vram:
			return self.x1[idx].to(torch.int), self.x2[idx].to(torch.float), torch.clip(self.y[idx].to(torch.float), min=-200, max=200)/200
		else:
			return self.x[idx][:64:].to("cuda").to(torch.int), self.x[idx][64::].to("cuda").to(torch.float), torch.clip(self.y[idx].to("cuda").to(torch.float), min=-200, max=200)/200


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
		return torch.sparse.mm(x, masked_weight.T) + self.bias

	
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
class Net_v5(nn.Module):#sparse layers and embeddings
	def __init__(self, mask1, mask2):
		super(Net_v5, self).__init__()
		self.embedding = nn.Embedding(13, 8)
		self.s = nn.Sequential(
			SparseConnectedLayer(534, 4096, mask1, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			SparseConnectedLayer(4096, 4096, mask2, 'leaky_relu', 0.01),
			nn.LeakyReLU(),
			nn.Linear(4096, 1024),
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


def worker_init_fn(worker_id):
	# Assign a global variable based on worker_id for data source behavior
	global data_in_vram_flag
	if worker_id == 0:
		data_in_vram_flag = True
		print(f"Worker {worker_id}: data_in_vram_flag = {data_in_vram_flag}")
	else:
		data_in_vram_flag = False
		print(f"Worker {worker_id}: data_in_vram_flag = {data_in_vram_flag}")



	
def calculate_mask1(embedding_dim):#embedding vocabluary size - 13 embeddig dimension - 8
	mask = torch.zeros(65, 534)
	for i in range(64):
		for j in range(6):
			mask[i][64*embedding_dim+j] = 1#turn and castling rights

	for i in range(64):
		x = i%8
		y = i//8
		mask[i][y*8*embedding_dim:(y+1)*8*embedding_dim:] = 1
		for j in range(8):
			mask[i][j*embedding_dim*8+x*embedding_dim:j*embedding_dim*8+x*embedding_dim+embedding_dim] = 1
		for j in range(8):
			for k in range(8):
				if j+k==x+y or j-k==x-y:
					mask[i][j*embedding_dim+k*embedding_dim*8:(j*embedding_dim+k*embedding_dim*8)+embedding_dim] = 1
		if x>1:
			if y>1:
				mask[i][(y-2)*8*embedding_dim+(x-1)*embedding_dim:(y-2)*8*embedding_dim+(x-1)*embedding_dim+embedding_dim] = 1
				mask[i][(y-1)*8*embedding_dim+(x-2)*embedding_dim:(y-1)*8*embedding_dim+(x-2)*embedding_dim+embedding_dim] = 1
			elif y>0:
				mask[i][(y-1)*8*embedding_dim+(x-2)*embedding_dim:(y-1)*8*embedding_dim+(x-2)*embedding_dim+embedding_dim] = 1
			if y<6:
				mask[i][(y+2)*8*embedding_dim+(x-1)*embedding_dim:(y+2)*8*embedding_dim+(x-1)*embedding_dim+embedding_dim] = 1
				mask[i][(y+1)*8*embedding_dim+(x-2)*embedding_dim:(y+1)*8*embedding_dim+(x-2)*embedding_dim+embedding_dim] = 1
			elif y<7:
				mask[i][(y+1)*8*embedding_dim+(x-2)*embedding_dim:(y+1)*8*embedding_dim+(x-2)*embedding_dim+embedding_dim] = 1
		elif x>0:
			if y>1:
				mask[i][(y-2)*8*embedding_dim+(x-1)*embedding_dim:(y-2)*8*embedding_dim+(x-1)*embedding_dim+embedding_dim] = 1
			if y<6:
				mask[i][(y+2)*8*embedding_dim+(x-1)*embedding_dim:(y+2)*8*embedding_dim+(x-1)*embedding_dim+embedding_dim] = 1
		if x<6:
			if y>1:
				mask[i][(y-2)*8*embedding_dim+(x+1)*embedding_dim:(y-2)*8*embedding_dim+(x+1)*embedding_dim+embedding_dim] = 1
				mask[i][(y-1)*8*embedding_dim+(x+2)*embedding_dim:(y-1)*8*embedding_dim+(x+2)*embedding_dim+embedding_dim] = 1
			elif y>0:
				mask[i][(y-1)*8*embedding_dim+(x+2)*embedding_dim:(y-1)*8*embedding_dim+(x+2)*embedding_dim+embedding_dim] = 1
			if y<6:
				mask[i][(y+2)*8*embedding_dim+(x+1)*embedding_dim:(y+2)*8*embedding_dim+(x+1)*embedding_dim+embedding_dim] = 1
				mask[i][(y+1)*8*embedding_dim+(x+2)*embedding_dim:(y+1)*8*embedding_dim+(x+2)*embedding_dim+embedding_dim] = 1
			elif y<7:
				mask[i][(y+1)*8*embedding_dim+(x+2)*embedding_dim:(y+1)*8*embedding_dim+(x+2)*embedding_dim+embedding_dim] = 1
		elif x<7:
			if y>1:
				mask[i][(y-2)*8*embedding_dim+(x+1)*embedding_dim:(y-2)*8*embedding_dim+(x+1)*embedding_dim+embedding_dim] = 1
			if y<6:
				mask[i][(y+2)*8*embedding_dim+(x+1)*embedding_dim:(y+2)*8*embedding_dim+(x+1)*embedding_dim+embedding_dim] = 1
	mask[64][64*embedding_dim:534] = 1
	return mask

def calculate_mask2():
	mask = torch.zeros(65, 4096)
	for i in range(64):
		for j in range(64):
			mask[i][64*63+j] = 1#turn and castling rights

	for i in range(64):
		x = i%8
		y = i//8
		mask[i][y*8*63:(y+1)*8*63:] = 1
		for j in range(8):
			mask[i][j*63*8+x*63:j*63*8+x*63+63] = 1
		for j in range(8):
			for k in range(8):
				if j+k==x+y or j-k==x-y:
					mask[i][j*63+k*63*8:(j*63+k*63*8)+63] = 1
		if x>1:
			if y>1:
				mask[i][(y-2)*8*63+(x-1)*63:(y-2)*8*63+(x-1)*63+63] = 1
				mask[i][(y-1)*8*63+(x-2)*63:(y-1)*8*63+(x-2)*63+63] = 1
			elif y>0:
				mask[i][(y-1)*8*63+(x-2)*63:(y-1)*8*63+(x-2)*63+63] = 1
			if y<6:
				mask[i][(y+2)*8*63+(x-1)*63:(y+2)*8*63+(x-1)*63+63] = 1
				mask[i][(y+1)*8*63+(x-2)*63:(y+1)*8*63+(x-2)*63+63] = 1
			elif y<7:
				mask[i][(y+1)*8*63+(x-2)*63:(y+1)*8*63+(x-2)*63+63] = 1
		elif x>0:
			if y>1:
				mask[i][(y-2)*8*63+(x-1)*63:(y-2)*8*63+(x-1)*63+63] = 1
			if y<6:
				mask[i][(y+2)*8*63+(x-1)*63:(y+2)*8*63+(x-1)*63+63] = 1
		if x<6:
			if y>1:
				mask[i][(y-2)*8*63+(x+1)*63:(y-2)*8*63+(x+1)*63+63] = 1
				mask[i][(y-1)*8*63+(x+2)*63:(y-1)*8*63+(x+2)*63+63] = 1
			elif y>0:
				mask[i][(y-1)*8*63+(x+2)*63:(y-1)*8*63+(x+2)*63+63] = 1
			if y<6:
				mask[i][(y+2)*8*63+(x+1)*63:(y+2)*8*63+(x+1)*63+63] = 1
				mask[i][(y+1)*8*63+(x+2)*63:(y+1)*8*63+(x+2)*63+63] = 1
			elif y<7:
				mask[i][(y+1)*8*63+(x+2)*63:(y+1)*8*63+(x+2)*63+63] = 1
		elif x<7:
			if y>1:
				mask[i][(y-2)*8*63+(x+1)*63:(y-2)*8*63+(x+1)*63+63] = 1
			if y<6:
				mask[i][(y+2)*8*63+(x+1)*63:(y+2)*8*63+(x+1)*63+63] = 1
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





def smooth_max(x, y, beta=10):
	return (1 / beta) * torch.log(torch.exp(beta * x) + torch.exp(beta * y))

class squareRootLoss(nn.Module):
	def __init__(self, epsilon=0.000001):
		super(squareRootLoss, self).__init__()
		self.epsilon = epsilon

	def forward(self, predictions, targets):
		loss = torch.sqrt(torch.sqrt((predictions - targets) ** 2+self.epsilon))
		return loss.mean()  # Return the mean loss across the batch

class FourthRootLossSmoothMax(nn.Module):
	def __init__(self, epsilon=0.000001):
		super(squareRootLoss, self).__init__()
		self.epsilon = epsilon

	def forward(self, predictions, targets):
		loss = torch.sqrt(torch.sqrt(smooth_max(torch.sqrt((predictions - targets) ** 2+self.epsilon), torch.tensor([0.01]))))-0.44821529903341381#0.44821529903341381 is the value of sqrt(sqrt(smooth_max(sqrt(0.000001), 0.01)))  subtructed to make the loss 0 when the difference between predictions and targets is 0
		return loss.mean()  # Return the mean loss across the batch


def main():
	
	if torch.cuda.is_available():
		print("\033[32mCUDA is available\033[0m\n\n")
		device = "cuda"
	else:
		device = "cpu"
		print("\033[31mCUDA is unavailable!\033[0m\n\n")


	compressed_mask1 = calculate_mask1(8)
	mask1 = torch.cat((compressed_mask1.unsqueeze(1).repeat(1, 63, 1).view(-1, compressed_mask1.size(1)), compressed_mask1[64].unsqueeze(0)), dim=0)
	#mask2 = calculate_mask2()
	compressed_mask2 = calculate_mask2()
	mask2 = torch.cat((compressed_mask2.unsqueeze(1).repeat(1, 63, 1).view(-1, compressed_mask2.size(1)) ,compressed_mask2[64].unsqueeze(0)), dim=0)
	

	batch_size = 1<<16

	input_bin_file_test = 'D:/private/chess engine/data/x_test.bin'
	label_bin_file_test = 'D:/private/chess engine/data/y_test.bin'

	input_size = 790
	label_size = 16
	test_elements_number = 1660516

	#test dataset
	input_bin_file_train = 'D:/private/chess engine/data/x2_train.bin'
	label_bin_file_train = 'D:/private/chess engine/data/y2_train.bin'
	input_bin_file_test = 'D:/private/chess engine/data/x2_test.bin'
	label_bin_file_test = 'D:/private/chess engine/data/y2_test.bin'

	input_size = 86
	label_size = 2
	train_elements_number = 81365220
	test_elements_number = 1660516

	#test dataset
	test_dataset = CustomDataset(input_bin_file_test, label_bin_file_test, test_elements_number, input_size)


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


	

	
	#net = Net_v4(mask1, mask2)
	net = Net_v5(mask1, mask2)
	saving_path = ""
	saving_path = "D:/private/chess engine/nets/2024-10-14_13-29-48"
	if saving_path=="":
		starting_program_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		saving_path = "D:/private/chess engine/nets/"+str(starting_program_time)

		os.makedirs(saving_path, exist_ok=True)

		iterations = 0
		with open(saving_path+"/iterations.txt", "w") as f:
			f.write(str(iterations))
	else:
		net.load_state_dict(torch.load(saving_path+"/4.pth")['model_state_dict'])
		#print loss
		print(torch.load(saving_path+"/4.pth")['test_loss'])


	
	net = net.to(device)

	#criterion = squareRootLoss()
	criterion = nn.MSELoss()


	
	abs_error_sum = 0
	relative_abs_error_sum = 0
	
	start_time = time.time()
	iterations=0


	"""
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
	"""
	with torch.inference_mode():
		net.eval()
		running_loss_test = 0.0
		for test_batch in test_dataloader:
			#batch_label = test_batch[1].to(device)
			output = net(test_batch[0], test_batch[1])

			
			loss = criterion(output, test_batch[2].to(device))
			running_loss_test += loss.item()
			# Move test_batch[1] to device once
			y_true = test_batch[2].to(device)
			y_pred = output

			# Apply condition: Only consider y_true values between -1 and 1
			condition = (y_true >= -1) & (y_true <= 1)

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
			for i in range(len(test_batch[1])):
				print(f"{(output[i].item()*200):.0f} {(test_batch[2][i].item()*200):.0f}")
			print("|", end="")
		
	print(f"test loss: {running_loss_test/(len(test_dataset)/batch_size)}")
	print(f"abs error: {abs_error_sum/(len(test_dataset))}")
	print(f"relative abs error: {relative_abs_error_sum/(len(test_dataset))}")
	input()


if __name__ == "__main__":
	main()