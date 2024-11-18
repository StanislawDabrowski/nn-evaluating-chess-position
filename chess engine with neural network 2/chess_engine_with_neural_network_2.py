import os
from random import shuffle
import torch
import torch.nn as nn
import time
#import matplotlib.pyplot as plt
from datetime import datetime
from bitarray import bitarray
from torch.utils.data import Dataset, DataLoader, dataset
import numpy as np
import threading



data_in_vram_flag = False

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
		if self.data_in_vram:
			return self.x1[idx].to(torch.int), self.x2[idx].to(torch.float), self.y[idx].to(torch.float)/10000
		else:
			return self.x[idx][:64:].to("cuda").to(torch.int), self.x[idx][64::].to("cuda").to(torch.float), self.y[idx].to("cuda").to(torch.float)/10000




class CustomDataLoader(DataLoader):
	def __iter__(self):
		self.dataset.data_in_vram = globals().get('data_in_vram_flag')
		return super().__iter__()


def worker_init_fn(worker_id):
	# Assign a global variable based on worker_id for data source behavior
	global data_in_vram_flag
	if worker_id == 0:
		if torch.cuda.is_available()=="cuda":
			data_in_vram_flag = True
		else:
			data_in_vram_flag = False
		print(f"Worker {worker_id}: data_in_vram_flag = {data_in_vram_flag}")
	else:
		data_in_vram_flag = False
		print(f"Worker {worker_id}: data_in_vram_flag = {data_in_vram_flag}")


def save_model(path, epoch, model_state_dict, optimizer_state_dict, test_loss):
	torch.save({
	'epoch': epoch,
	'model_state_dict': model_state_dict(),
	'optimizer_state_dict': optimizer_state_dict(),
	"""'scheduler_state_dict': scheduler_state_dict(),"""
	'test_loss': test_loss,
	}, path)






def differentiable_abs(a, epsilon=0.000001):
	return torch.sqrt(a ** 2 + epsilon)
	

class SquareRootCubeLoss(nn.Module):
	def __init__(self, epsilon=0.000001):
		super(SquareRootCubeLoss, self).__init__()
		self.epsilon = epsilon

	def forward(self, predictions, targets):
		loss = torch.sqrt(differentiable_abs(predictions - targets) ** 3+self.epsilon)
		return loss.mean()


class SignedlMSE(nn.Module):
	def __init__(self, alpha=3, epsilon=0.01):
		super(SignedlMSE, self).__init__()
		self.epsilon = epsilon
		self.alpha = alpha

	def forward(self, predictions, targets):
		difference = predictions - targets
		targets_abs = torch.abs(targets)
		loss = torch.where((targets*predictions)>0, difference*difference, self.alpha*(difference*difference)-((self.alpha-1)*targets*targets))
		return loss.mean()


class ProportionalSignedlMSE(nn.Module):
	def __init__(self, alpha=2, epsilon=0.01):
		super(ProportionalSignedlMSE, self).__init__()
		self.epsilon = epsilon
		self.alpha = alpha

	def forward(self, predictions, targets):
		difference = predictions - targets
		targets_abs = torch.abs(targets)
		loss = torch.where((targets*predictions)>0, difference*difference/(targets_abs+self.epsilon), self.alpha*(difference*difference)/(targets_abs+self.epsilon)-((self.alpha-1)*targets*targets/(targets_abs+self.epsilon)))
		return loss.mean()


	
class SparseConnectedLayer(nn.Module):
	def __init__(self, in_features, out_features, mask, nonlinearity, negative_slope=0):
		super(SparseConnectedLayer, self).__init__()
		self.weight = nn.Parameter(torch.zeros(out_features, in_features))
		torch.nn.init.kaiming_normal_(self.weight, a=negative_slope, nonlinearity=nonlinearity)
		self.bias = nn.Parameter(torch.zeros(out_features))
		self.register_buffer('mask', mask)
    
	def forward(self, x):
		masked_weight = self.weight * self.mask
		return torch.sparse.mm(x, masked_weight.T) + self.bias

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
		x1 = self.embedding(x1)
		x1 = x1.view(x1.size(0), -1)
		x = torch.cat((x1, x2), 1)
		return self.s(x)




def calculate_mask1(embedding_dim):#embedding vocabluary size - 13 embeddig dimension - 8
	mask = torch.zeros(65, 64*embedding_dim+22)
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
	mask[64][64*embedding_dim:64*embedding_dim+22] = 1
	return mask


def calculate_mask2(mask1_size, parameters_number):
	mask = torch.zeros(65, mask1_size)
	for i in range(64):
		for j in range(64*parameters_number, mask1_size):
			mask[i][j] = 1#turn and castling rights

	for i in range(64):
		x = i%8
		y = i//8
		mask[i][y*8*parameters_number:(y+1)*8*parameters_number:] = 1
		for j in range(8):
			mask[i][j*parameters_number*8+x*parameters_number:j*parameters_number*8+x*parameters_number+parameters_number] = 1
		for j in range(8):
			for k in range(8):
				if j+k==x+y or j-k==x-y:
					mask[i][j*parameters_number+k*parameters_number*8:(j*parameters_number+k*parameters_number*8)+parameters_number] = 1
		if x>1:
			if y>1:
				mask[i][(y-2)*8*parameters_number+(x-1)*parameters_number:(y-2)*8*parameters_number+(x-1)*parameters_number+parameters_number] = 1
				mask[i][(y-1)*8*parameters_number+(x-2)*parameters_number:(y-1)*8*parameters_number+(x-2)*parameters_number+parameters_number] = 1
			elif y>0:
				mask[i][(y-1)*8*parameters_number+(x-2)*parameters_number:(y-1)*8*parameters_number+(x-2)*parameters_number+parameters_number] = 1
			if y<6:
				mask[i][(y+2)*8*parameters_number+(x-1)*parameters_number:(y+2)*8*parameters_number+(x-1)*parameters_number+parameters_number] = 1
				mask[i][(y+1)*8*parameters_number+(x-2)*parameters_number:(y+1)*8*parameters_number+(x-2)*parameters_number+parameters_number] = 1
			elif y<7:
				mask[i][(y+1)*8*parameters_number+(x-2)*parameters_number:(y+1)*8*parameters_number+(x-2)*parameters_number+parameters_number] = 1
		elif x>0:
			if y>1:
				mask[i][(y-2)*8*parameters_number+(x-1)*parameters_number:(y-2)*8*parameters_number+(x-1)*parameters_number+parameters_number] = 1
			if y<6:
				mask[i][(y+2)*8*parameters_number+(x-1)*parameters_number:(y+2)*8*parameters_number+(x-1)*parameters_number+parameters_number] = 1
		if x<6:
			if y>1:
				mask[i][(y-2)*8*parameters_number+(x+1)*parameters_number:(y-2)*8*parameters_number+(x+1)*parameters_number+parameters_number] = 1
				mask[i][(y-1)*8*parameters_number+(x+2)*parameters_number:(y-1)*8*parameters_number+(x+2)*parameters_number+parameters_number] = 1
			elif y>0:
				mask[i][(y-1)*8*parameters_number+(x+2)*parameters_number:(y-1)*8*parameters_number+(x+2)*parameters_number+parameters_number] = 1
			if y<6:
				mask[i][(y+2)*8*parameters_number+(x+1)*parameters_number:(y+2)*8*parameters_number+(x+1)*parameters_number+parameters_number] = 1
				mask[i][(y+1)*8*parameters_number+(x+2)*parameters_number:(y+1)*8*parameters_number+(x+2)*parameters_number+parameters_number] = 1
			elif y<7:
				mask[i][(y+1)*8*parameters_number+(x+2)*parameters_number:(y+1)*8*parameters_number+(x+2)*parameters_number+parameters_number] = 1
		elif x<7:
			if y>1:
				mask[i][(y-2)*8*parameters_number+(x+1)*parameters_number:(y-2)*8*parameters_number+(x+1)*parameters_number+parameters_number] = 1
			if y<6:
				mask[i][(y+2)*8*parameters_number+(x+1)*parameters_number:(y+2)*8*parameters_number+(x+1)*parameters_number+parameters_number] = 1
	mask[64][::] = 1
	return mask


def main():

	
	if torch.cuda.is_available():
		print("\033[32mCUDA is available\033[0m\n\n")
		device = "cuda"
	else:
		device = "cpu"
		print("\033[31mCUDA is unavailable!\033[0m\n\n")
		
	batch_size = int(input())

	

	input_bin_file_train = '../files/data/x2_train.bin'
	label_bin_file_train = '../files/data/y2_train.bin'
	input_bin_file_test = '../files/data/x2_test.bin'
	label_bin_file_test = '../files/data/y2_test.bin'

	input_size = 86
	label_size = 2
	train_elements_number = 81365220
	test_elements_number = 1660516

	#train dataset
	dataset = CustomDataset(input_bin_file_train, label_bin_file_train, train_elements_number, input_size)
	#test dataset
	test_dataset = CustomDataset(input_bin_file_test, label_bin_file_test, test_elements_number, input_size)


	#train dataloader
	dataloader = CustomDataLoader(dataset, batch_size=batch_size, num_workers=3, worker_init_fn=worker_init_fn, shuffle=True)
	#test dataloader
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


	print("succesfuly loaded data")
	os.system("cls")



	


	compressed_mask1 = calculate_mask1(6)
	mask1 = torch.cat((compressed_mask1.unsqueeze(1).repeat(1, 31, 1).view(-1, compressed_mask1.size(1)), compressed_mask1[64].unsqueeze(0).repeat(33, 1)), dim=0)

	compressed_mask2 = calculate_mask2(2048, 31)
	mask2 = torch.cat((compressed_mask2.unsqueeze(1).repeat(1, 31, 1).view(-1, compressed_mask2.size(1)) ,compressed_mask2[64].unsqueeze(0).repeat(33, 1)), dim=0)
	
	

	




	net = Net_v6(mask1, mask2)
	saving_path = ""
	#saving_path = "../nets/2024-10-31_13-58-41/"
	if saving_path=="":
		starting_program_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		saving_path = "../files/nets/"+str(starting_program_time)

		os.makedirs(saving_path, exist_ok=True)

		iterations = 0
		with open(saving_path+"/iterations.txt", "w") as f:
			f.write(str(iterations))
	else:
		net.load_state_dict(torch.load(saving_path+"8.pth")['model_state_dict'])


	
	net = net.to(device)

	
	epochs = 10

	#mean absolute errors loss
	#criterion = SquareRootCubeLoss()
	criterion = SignedlMSE()
	#criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=2e-7)

	def change_learning_rate():
		while True:
			try:
				new_lr = input()
				if new_lr:
					new_lr = float(new_lr)
					for param_group in optimizer.param_groups:
						param_group['lr'] = new_lr
					print(f"Learning rate updated to {new_lr}")
			except ValueError:
				print("Invalid input")
			except KeyboardInterrupt:
				break

	# Start a separate thread to listen for learning rate changes
	thread = threading.Thread(target=change_learning_rate)
	thread.start()



	#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
		"""f.write(f"scheduler_type: {type(scheduler).__name__}\n")"""

	for epoch in range(epochs):
		print(epoch)
		batches_proceded = 0
		for batch in dataloader:
			if batches_proceded%((train_elements_number/batch_size)//1000)==0:
				print(f"batches preceded: {batches_proceded} {(batches_proceded/((train_elements_number/batch_size)//100)):.2f}% {(time.time()-start_time):.2f} seconds", end="")
				if batches_proceded!=0:
					print(f" loss: {loss.item()}")
				else:
					print()
				start_time = time.time()
			optimizer.zero_grad()
			output = net(batch[0], batch[1])
			loss = criterion(output, batch[2])
			running_loss += loss.item()

			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())
			batches_proceded+=1
		if epoch%testing_iterations==0:
			with torch.inference_mode():
				net.eval()
				running_loss_test = 0.0
				ite = 0
				for test_batch in test_dataloader:
					if ite%100==0 and ite!=0:
						#print(ite, end=" ", flush=True)
						print(running_loss_test/ite)
					#print()
					output = net(test_batch[0], test_batch[1])
					loss = criterion(output, test_batch[2])
					running_loss_test += loss.item()
					ite+=1
				test_losses.append(running_loss_test/(len(test_dataset)/batch_size))
				print(f"test loss: {running_loss_test/(len(test_dataset)/batch_size)}")
				#train loss
				print(f"train loss: {running_loss/(len(dataset)/batch_size)}")
				running_loss = 0.0
				net.train()
		if epoch%saving_iterations==0:
			iterations+=saving_iterations
			save_model(saving_path+"/"+str(iterations)+".pth", epoch, net.state_dict, optimizer.state_dict, test_losses[-1])
			with open(saving_path+"/iterations.txt", "w") as f:
				f.write(str(iterations))


	"""
	#vusialize the loss on a plot
	plt.plot(train_losses, label="train")
	plt.plot(test_losses, label="test")
	plt.legend()
	plt.show()
	"""

if __name__ == "__main__":
	main()