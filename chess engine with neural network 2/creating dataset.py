
#creating dataset

'''
import time


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
#x_temp = bitarray(790*4)
x_temp = [0 for _ in range(86)]


"""
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
"""

def get_piece_index2(piece):
	if piece == "P":
		return 1
	if piece == "N":
		return 2
	if piece == "B":
		return 3
	if piece == "R":
		return 4
	if piece == "Q":
		return 5
	if piece == "K":
		return 6
	if piece == "p":
		return 7
	if piece == "n":
		return 8
	if piece == "b":
		return 9
	if piece == "r":
		return 10
	if piece == "q":
		return 11
	if piece == "k":
		return 12
	return -1


start_time = time.time()


with open("D:/private/chess engine/data/x2.bin", "ab") as f:
	for i in range(len(positions)):
		if i%100000==0:
			print(f"converting positions {i/len(positions)*100}%   " + str(time.time() - start_time) + " seconds")
			start_time = time.time()
		position = positions[i].split(" ")
		board = position[0]
		rank = 0
		file = 0
		for j in range(len(board)):
			if board[j].isdigit():
				empty = int(board[j])
				for k in range(empty):
					x_temp[rank*8+file+k]=0
				file+=empty
			elif board[j] == "/":
				rank+=1
				file = 0
			else:
				x_temp[rank*8+file]=get_piece_index2(board[j])
				file+=1
			move = position[1]
			if move=="w":
				x_temp[64]=1
			else:
				x_temp[65]=1
			castle = position[2]
			if "K" in castle:
				x_temp[66]=1
			if "Q" in castle:
				x_temp[67]=1
			if "k" in castle:
				x_temp[68]=1
			if "q" in castle:
				x_temp[69]=1
			en_passant = position[3]
			if en_passant!="-":
				x_temp[70+ord(en_passant[0])-97 + 8*((int(en_passant[1])-3)//3)] = 1#if en_passant[1]=="6" rank is 1 else it's 0 ( (3-3)/3==0 and (6-3)/3==1 )
			#save as bytes (0-255) values in .bin file
		f.write(bytes(x_temp))
		x_temp = [0 for _ in range(86)]



'''
'''
chunk_index = 0

with open("D:/private/chess engine/data/x2.bin", "ab") as f:
	for i in range(len(positions)):
		if i%100000==0:
			print(f"converting positions {i/len(positions)*100}%   " + str(time.time() - start_time) + " seconds")
			start_time = time.time()
		position = positions[i].split(" ")
		board = position[0]
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
#splitting files

import time

def split_file_in_chunks(input_file, output_file1, output_file2, n, chunk_size):
	'''
	:param input_file: path to input file
	:param output_file1: path to output file1
	:param output_file2: path to output file2
	:param n: number of bytes to write to output_file1, total size-n odesn't need to be full bytes
	:param chunk_size: size of one chunk in bytes
	'''
	bytes_remaining = n

    # Open the input file in binary read mode
	with open(input_file, 'rb') as infile, open(output_file1, 'wb') as outfile1, open(output_file2, 'wb') as outfile2:
		while bytes_remaining > 0:
            # Read the minimum of chunk_size or remaining bytes
			bytes_to_read = min(chunk_size, bytes_remaining)
			data = infile.read(bytes_to_read)

            # Write the chunk to the output file
			outfile1.write(data)

			# Decrease the remaining byte count
			bytes_remaining -= bytes_to_read
		#while end of file is not reached
		while data:
			data = infile.read(chunk_size)
			outfile2.write(data)




def split_files(x_file_path, x_train_file_path, x_test_file_path, y_file_path, y_train_file_path, y_test_file_path, x_value_size, y_value_size, train_elements, test_elements):
	'''
	:param x_file_path: path to x file
	:param x_train_file_path: path to x train file
	:param x_test_file_path: path to x test file
	:param y_file_path: path to y file
	:param y_train_file_path: path to y train file
	:param y_test_file_path: path to y test file
	:param x_value_size: size of one x value in bits
	:param y_value_size: size of one y value in bits
	:param train_elements: number of train elements
	:param test_elements: number of test elements
	'''
	if (test_elements*x_value_size)%8!=0:
		raise Exception("test_elements*x_value_size must be dividable by 8")
	start_time = time.time()
	split_file_in_chunks(y_file_path, y_test_file_path, y_train_file_path, int(test_elements*y_value_size/8), 1<<20)
	print("y split time: ", time.time()-start_time)
	split_file_in_chunks(x_file_path, x_test_file_path, x_train_file_path, int(test_elements*x_value_size/8), 1<<30)




	
#train elements: 81365223   81365224 - teoretical proper value - changed to ajust to the test number dividable by 4 to facilitate spliting files (enables reading file with bytes insted of bits)
#test elements:   1660516    1660515 - teoretical proper value - changed to ajust to the test number dividable by 4 to facilitate spliting files (enables reading file with bytes insted of bits)
#            D:\private\chess engine\data
split_files("D:/private/chess engine/data/x2.bin", "D:/private/chess engine/data/x2_train.bin", "D:/private/chess engine/data/x2_test.bin", "D:/private/chess engine/data/y2.bin", "D:/private/chess engine/data/y2_train.bin", "D:/private/chess engine/data/y2_test.bin", 86*8, 16, 81365223, 1660516)
"""

"""
def delete_bytes(file_path, bytes_to_delete):
	with open(file_path, 'rb') as file:
		data = file.read()
	with open(file_path, 'wb') as file:
		file.write(data[:len(data)-bytes_to_delete])

delete_bytes("D:/private/chess engine/data/x2_train.bin", 3*86)
"""
