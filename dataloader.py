import torch
import torch.utils.data as data
import numpy as np
import random

random.seed(1143)


class noises_loader(data.Dataset):

	def __init__(self):
		self.noise_list = []
		self.V_list = []
		
		for i in range(2000): 
			noise = np.around(np.random.normal(0, 1, (128, 128, 3)) * 255)
			max_value = np.max(noise)
			min_value = np.min(noise)
			noise = (noise - min_value) / (max_value - min_value)
			self.noise_list.append(noise)
			
		print("Total training examples:", len(self.noise_list))

		for i in range(2000):
			V = np.random.random() * 1.5 - 0.5  # [-0.5, 1.0]
			self.V_list.append(V)


	def __getitem__(self, index):
		data_noise = self.noise_list[index]
		data_noise = (np.asarray(data_noise))
		data_noise = torch.from_numpy(data_noise).float()
		data_noise = data_noise.permute(2,0,1)
		data_V = self.V_list[index]

		return data_noise, data_V


	def __len__(self):
		return len(self.noise_list)
