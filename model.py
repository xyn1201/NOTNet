import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class enhance_net(nn.Module):

	def __init__(self):
		super(enhance_net, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

		self.fc1 = nn.Linear(1, 32)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True, padding_mode='replicate') 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True, padding_mode='replicate') 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True, padding_mode='replicate') 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True, padding_mode='replicate') 
		self.e_conv5 = nn.Conv2d(number_f,number_f,3,1,1,bias=True, padding_mode='replicate') 
		self.e_conv6 = nn.Conv2d(number_f,number_f,3,1,1,bias=True, padding_mode='replicate') 
		self.e_conv7 = nn.Conv2d(number_f,32,3,1,1,bias=True, padding_mode='replicate') 

		self.e_conv_final = nn.Conv2d(1,6,3,1,1,bias=True, padding_mode='replicate') 


		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x, V):

		b, _, h, w = x.shape

		W = self.relu(self.fc1(V))

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(x4))
		x6 = self.relu(self.e_conv6(x5))
		x_r = self.e_conv7(x6)
	
		x_r = F.conv2d(x_r.view(1, b * 32, h, w),
				W.view(b, 32, 1, 1), groups=b)
		
		x_r = x_r.view(b, 1, h, w)
		x_r = self.e_conv_final(x_r) 


		# linear curve
		k, b = torch.split(x_r, 3, dim=1)
		k = self.sigmoid(k) * 16
		b = self.sigmoid(b) * 0.2

		enhanced_image = k * x + b
		enhanced_image = torch.clip(enhanced_image, 0, 1)
		

		return enhanced_image
	