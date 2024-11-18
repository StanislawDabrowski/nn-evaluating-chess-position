import torch
import time





'''
t = torch.zeros(81000000, 100, dtype=torch.bool).to('cuda')

start_time = time.time()

for i in range(79*1024):
	#t2 = t[i][:64:].to('cuda').to(torch.int)
	#t2 = t[i][64::].to('cuda').to(torch.int)
	t2 = t[i][:64:]
	t2 = t[i][64::]
	#t2 = t[i][:64:].to('cuda')
	#t2 = t[i][64::].to('cuda')
	#t2 = t[i][:64:].to(torch.int)
	#t2 = t[i][64::].to(torch.int)




print(time.time() - start_time)
'''

'''
import bitarray


ba = bitarray.bitarray(8100*22)
ba.setall(0)
print("created")

start_time = time.time()
for i in range(1000):
	bool_list = ba.tolist()

	tensor_bool = torch.tensor(bool_list, dtype=torch.float)

print(time.time() - start_time)
input()
'''

'''
t = torch.zeros(100, 100, 100)

start_time = time.time()
for i in range(1000000):
	t.view(1000, 10, 100)

print(time.time() - start_time)
input()
'''