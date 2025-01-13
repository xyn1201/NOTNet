import torch
import torch.nn as nn
import torch.nn.functional as F


class L_spa(nn.Module):

	def __init__(self):
		super(L_spa, self).__init__()
		
		kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
		kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
		kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
		kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
		self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
		self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
		self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
		self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
		self.pool = nn.AvgPool2d(4)
		

	def forward(self, org , enhance):
		b,c,h,w = org.shape

		org_mean = torch.mean(org,1,keepdim=True)
		enhance_mean = torch.mean(enhance,1,keepdim=True)

		org_pool =  self.pool(org_mean)			
		enhance_pool = self.pool(enhance_mean)	

		D_org_left = F.conv2d(org_pool , self.weight_left, padding=1)
		D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
		D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
		D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

		D_enhance_left = F.conv2d(enhance_pool , self.weight_left, padding=1)
		D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
		D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
		D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

		D_left = torch.pow(D_org_left - D_enhance_left,2)
		D_right = torch.pow(D_org_right - D_enhance_right,2)
		D_up = torch.pow(D_org_up - D_enhance_up,2)
		D_down = torch.pow(D_org_down - D_enhance_down,2)
		E = D_left + D_right + D_up + D_down

		return E