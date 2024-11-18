import os
import torch
import torch.nn as nn
import time
#import matplotlib.pyplot as plt
from datetime import datetime
from bitarray import bitarray
import numpy as np


def save_model(path, epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, test_loss):
	torch.save({
	'epoch': epoch,
	'model_state_dict': model_state_dict(),
	'optimizer_state_dict': optimizer_state_dict(),
	'scheduler_state_dict': scheduler_state_dict(),
	'test_loss': test_loss,
	}, path)



class TestDataloader():
	def __init__(self, x, y, number_samples, batch_size):
		self.x = x
		self.y = y
		self.number_samples = number_samples
		self.batch_size = batch_size
	def __len__(self):
		return self.number_samples
	def __getbatch__(self, idx):
		if self.batch_size*(idx+1)<self.number_samples:
			return self.x[self.batch_size*idx:self.batch_size*(idx+1),:64:].to('cuda').to(torch.int), self.x[self.batch_size*idx:self.batch_size*(idx+1),64::].to('cuda').to(torch.float), self.y[self.batch_size*idx:self.batch_size*(idx+1)].to('cuda').to(torch.float)
		return self.x[self.batch_size*idx::,:64:].to('cuda').to(torch.int), self.x[self.batch_size*idx::,64::].to('cuda').to(torch.float), self.y[self.batch_size*idx::].to('cuda').to(torch.float)

	
class TrainDataloader():
	def __init__(self, x1, x2, y, number_samples, batch_size):
		self.x1 = x1.to('cuda')
		self.x2 = x2#bitarray
		self.y = y
		self.number_samples = number_samples
		self.batch_size = batch_size
	def __len__(self):
		return self.number_samples
	def __getbatch__(self, idx):
		if self.batch_size*(idx+1)<self.number_samples:
			x2_list = self.x2[self.batch_size*idx*22:self.batch_size*(idx+1)*22].tolist()
			x2 = torch.tensor(x2_list, dtype=torch.float).view(self.batch_size, 22).to('cuda')
			return self.x1[self.batch_size*idx:self.batch_size*(idx+1)].to(torch.int), x2, self.y[self.batch_size*idx:self.batch_size*(idx+1)].to('cuda').to(torch.float)
		x2_list = self.x2[self.batch_size*idx*22::].tolist()
		x2 = torch.tensor(x2_list, dtype=torch.float).view(self.number_samples-self.batch_size*idx, 22).to('cuda')
		return self.x1[self.batch_size*idx::].to(torch.int), self.x2, self.y[self.batch_size*idx::].to('cuda').to(torch.float)

def differentiable_abs(a, epsilon=0.000001):
	return torch.sqrt(a ** 2 + epsilon)
	

class SquareRootCubeLoss(nn.Module):
	def __init__(self, epsilon=0.000001):
		super(SquareRootCubeLoss, self).__init__()
		self.epsilon = epsilon

	def forward(self, predictions, targets):
		absv = differentiable_abs(predictions - targets, self.epsilon)
		loss = torch.sqrt(absv*absv*absv+self.epsilon)
		return loss.mean()
	
class SparseConnectedLayer(nn.Module):
	def __init__(self, in_features, out_features, mask, nonlinearity, negative_slope=0):
		super(SparseConnectedLayer, self).__init__()
		#initialize weights using torch.nn.init.kaiming_normal_()
		self.weight = nn.Parameter(torch.randn(out_features, in_features))
		#torch.nn.init.kaiming_normal_(self.weight, a=negative_slope, nonlinearity=nonlinearity)
		self.bias = nn.Parameter(torch.zeros(out_features))
		self.register_buffer('mask', mask)
    
	def forward(self, x):
		masked_weight = self.weight * self.mask
		return torch.sparse.mm(x, masked_weight.T) + self.bias

	
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



def main():

	
	if torch.cuda.is_available():
		print("\033[32mCUDA is available\033[0m\n\n")
		device = "cuda"
	else:
		device = "cpu"
		print("\033[31mCUDA is unavailable!\033[0m\n\n")
		
	batch_size_train = int(input())
	batch_size_test = 1<<10


	input_bin_file_train = 'D:/private/chess engine/data/x2_train.bin'
	label_bin_file_train = 'D:/private/chess engine/data/y2_train.bin'
	input_bin_file_test = 'D:/private/chess engine/data/x2_test.bin'
	label_bin_file_test = 'D:/private/chess engine/data/y2_test.bin'

	input_size = 86
	label_size = 2
	train_elements_number = 81365220
	test_elements_number = 1660516

	x_bytes_values = input_size
	x_path_train = input_bin_file_train
	samples_number_train = train_elements_number
	with open(label_bin_file_train, "rb") as f:
		y_bytes_train = f.read()
	y = torch.frombuffer(y_bytes_train, dtype=torch.int16).view(samples_number_train, 1)
	y_bytes_train = None

	
	
	with open(x_path_train, "rb") as f:
		x_bytes_train = f.read()
	byte_array_train_np = np.frombuffer(x_bytes_train, dtype=np.uint8)

	# Reshape the array to (n, 86)
	reshaped_array_train = byte_array_train_np.reshape(samples_number_train, x_bytes_values)

	# Split the reshaped array into two tensors
	x1 = torch.tensor(reshaped_array_train[:, :64:], dtype=torch.uint8)   # First 64 bytes
	x2 = torch.tensor(reshaped_array_train[:, 64::], dtype=torch.bool)    # Last 22 bytes

	#convert x2 to bitarray
	x2 = x2.view(-1)
	x2 = bitarray(x2.tolist())
	torch.cuda.empty_cache()
	
	print(x1.element_size() * x1.numel())

	x_bytes = None
	
	

	x_path_test = input_bin_file_test
	samples_number_test = test_elements_number
	with open(label_bin_file_test, "rb") as f:
		y_bytes_test = f.read()
	y_test = torch.frombuffer(y_bytes_test, dtype=torch.int16).view(samples_number_test, 1)
	y_bytes_test = None

	
	with open(x_path_test, "rb") as f:
		x_bytes_test = f.read()
	x_test = torch.frombuffer(x_bytes_test, dtype=torch.uint8).view(samples_number_test, x_bytes_values)
	x_bytes = None


	train_dataloader = TrainDataloader(x1, x2, y, train_elements_number, batch_size_train)
	test_dataloader = TestDataloader(x_test, y_test, test_elements_number, 1<<10)

	batches_numebr_train = train_elements_number//batch_size_train+1
	batches_numebr_test = test_elements_number//batch_size_test+1
	
	
	print("succesfuly loaded data")
	#time.sleep(2)
	os.system("cls")



	



	compressed_mask1 = calculate_mask1(8)
	mask1 = torch.cat((compressed_mask1.unsqueeze(1).expand(compressed_mask1.unsqueeze(1).size(0), 63*compressed_mask1.unsqueeze(1).size(1), compressed_mask1.unsqueeze(1).size(2)).reshape(-1, compressed_mask1.size(1)), compressed_mask1[64].unsqueeze(0)), dim=0)
	#mask2 = calculate_mask2()
	compressed_mask2 = calculate_mask2()
	mask2 = torch.cat((compressed_mask2.unsqueeze(1).expand(compressed_mask2.unsqueeze(1).size(0), 63*compressed_mask2.unsqueeze(1).size(1), compressed_mask2.unsqueeze(1).size(2)).reshape(-1, compressed_mask2.size(1)) ,compressed_mask2[64].unsqueeze(0)), dim=0)
	
	#print(mask1.size())
	#print(mask2.size())
	
	"""
	for i in range(64):
		print_board_from_mask(compressed_mask[i])
		print("\n")
	
	input()
	"""
			




	net = Net_v5(mask1, mask2)
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
	criterion = SquareRootCubeLoss()
	#criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.00001)



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
		f.write(f"batch_size: {batch_size_train}\n")
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
		overall = 0
		for name, param in net.named_parameters():
			overall+=param.numel()
			memory_in_bytes = param.element_size() * param.numel()
			print(f"Memory used by {name}: {memory_in_bytes} bytes")
		print(f"Overall memory used: {overall*4} bytes")
		if epoch%testing_iterations==0 and epoch>0:
			with torch.inference_mode():
				net.eval()
				running_loss_test = 0.0
				for test_batch_index in range(batches_numebr_test):
					test_batch = test_dataloader.__getbatch__(test_batch_index)
					if test_batch_index%10==0:
						print(test_batch_index/batches_numebr_test, end=" ", flush=True)
					output = net(test_batch[0], test_batch[1])
					loss = criterion(output, test_batch[2])
					running_loss_test += loss.item()
				test_losses.append(running_loss_test/(test_dataloader.__len__()/batch_size_test))
				print(f"test loss: {running_loss_test/(test_dataloader.__len__()/batch_size_test)}")
				#train loss
				print(f"train loss: {running_loss/(train_dataloader.__len__()/batch_size_train)}")
				running_loss = 0.0
				net.train()
		batches_preceded = 0
		for batch_index in range(batches_numebr_train):
			if batches_preceded%((train_elements_number/batch_size_train)//1000)==0:
				print(f"batches preceded: {batches_preceded} {(batches_preceded/((train_elements_number/batch_size_train)//100)):.2f}% {(time.time()-start_time):.2f} seconds")
				start_time = time.time()
			batch = train_dataloader.__getbatch__(batch_index)
			optimizer.zero_grad()
			output = net(batch[0], batch[1])
			loss = criterion(output, batch[2])
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
	#vusialize the loss on a plot
	plt.plot(train_losses, label="train")
	plt.plot(test_losses, label="test")
	plt.legend()
	plt.show()
	"""

if __name__ == "__main__":
	main()