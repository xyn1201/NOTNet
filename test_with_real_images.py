import torch
import torchvision
import torch.optim
import numpy as np
from PIL import Image
import glob
import os
import argparse
import model


def lowlight(image_path):
	
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	
	real_image_input = Image.open(image_path)
	real_image_input = (np.asarray(real_image_input)/255.0)
	real_image_input = torch.from_numpy(real_image_input).float()
	real_image_input = real_image_input.permute(2,0,1)
	real_image_input = real_image_input.cuda().unsqueeze(0)


	V = np.array([config.V])
	V = torch.from_numpy(V).float()
	V = V.cuda()


	NOTNet = model.enhance_net().cuda()
	NOTNet.load_state_dict(torch.load('snapshots/Epoch4.pth'))


	enhanced_image = NOTNet(real_image_input, V)


	enhanced_image_path = image_path.replace('test_data','test_enhanced')

	if not os.path.exists(enhanced_image_path.replace('/'+enhanced_image_path.split("/")[-1],'')):
		os.makedirs(enhanced_image_path.replace('/'+enhanced_image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, enhanced_image_path)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--V', type=float, default=0.4)
	config = parser.parse_args()

	with torch.no_grad():
		filePath = 'data/test_data/'
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = os.listdir(filePath + file_name + "/.")
			for image in test_list:
				full_image_path = filePath + file_name + "/" + image
				print(full_image_path)
				lowlight(full_image_path)

		


