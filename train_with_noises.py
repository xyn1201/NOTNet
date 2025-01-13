import torch
import torch.nn as nn
import torch.optim
import numpy as np
import os
import argparse
import dataloader
import model
import losses


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)



def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	NOTNet = model.enhance_net().cuda()
	NOTNet.apply(weights_init)
	if config.load_pretrain == True:
		NOTNet.load_state_dict(torch.load(config.pretrain_dir))
		
	train_dataset = dataloader.noises_loader()	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

	L_spa = losses.L_spa()

	optimizer = torch.optim.Adam(NOTNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	NOTNet.train()

	for epoch in range(config.num_epochs):
		for iteration, (noise_input, V) in enumerate(train_loader):

			noise_input = noise_input.cuda()
			V_img = np.ones((128, 128, 3)) * np.array(V)
			V_img = (np.asarray(V_img))
			V_img = torch.from_numpy(V_img).float()
			V_img = V_img.permute(2,0,1)
			V_img = V_img.cuda()
			V = V.cuda().to(torch.float32)
			

			# two references for contrast adjustment (both are ok, here we use the first one)
			# 1 adjust the dynamic range of noises
			noise_pseudoGT = (noise_input - 0.5) * (0.4) + 0.5
			# 2 adjust the kurtosis of noises
			# noise_pseudoGT = torch.clip(torch.log(noise_input / (1 - noise_input + 0.0001) + 0.0001) / (200) + 0.5, 0, 1)

			# mean-value adjustment
			noise_pseudoGT += V_img


			enhanced_image  = NOTNet(noise_input, V.cuda().to(torch.float32))

			
			loss_spa = torch.mean(L_spa(enhanced_image, noise_input))
			loss_rec = nn.L1Loss()(enhanced_image, noise_pseudoGT)
			loss = 2 * loss_rec + 6 * loss_spa 


			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(NOTNet.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Epoch ", epoch, " Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				torch.save(NOTNet.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=5)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=100)
	parser.add_argument('--snapshot_iter', type=int, default=100)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default=False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch4.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)
