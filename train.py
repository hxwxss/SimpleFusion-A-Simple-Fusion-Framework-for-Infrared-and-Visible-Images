import os
import scipy.io as scio
import torch
from torch.optim import Adam
from torch.autograd import Variable
from visdom import Visdom
from net import great_NET, Vgg16
from args import Args as args
import utils
import random
import datetime
from tqdm import tqdm
import time
import torch.nn as nn


import pdb


EPSILON = 1e-5
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w


def load_data(path, train_num):
	imgs_path, _ = utils.list_images(path)
	imgs_path = imgs_path[:train_num]
	random.shuffle(imgs_path)
	return imgs_path


def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss

def C_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2) 
    return loss

def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1) 
    loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2

def P_loss(im1, X1):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss

def retinex_loss_fuction(Lx, Rx, Ry, Nx ,batch_ir_in):
	
	loss1 = C_loss(Rx, Ry)
	loss2 = R_loss(Lx, Rx, batch_ir_in, Nx)
	loss3 = P_loss(batch_ir_in, Nx)
	loss =  loss1 * 1 + loss2 * 1 + loss3 * 500

	return loss


def train(data,lam2_vi,w_ir,w_vi,lam3_gram):

	img_flag = args.img_flag
	cuda_flag = args.cuda_flag
	batch_size = args.batch_size
	lr = args.lr
	vgg_model_dir = args.vgg_model_dir
	save_dir = args.save_dir
	num = args.num

	if args.img_flag == True:
		in_channels = 3
	else:
		in_channels = 1

	# creating save path
	current_time = datetime.datetime.now()
	formatted_time = current_time.strftime("%Y%m%d%H%M%S")

	save_dir = save_dir + "/"+ formatted_time 
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	mesg = "lam2 {}\t w_ir {}\t w_vi {}\t lam3 {}\t \n".format(lam2_vi, w_ir, w_vi, lam3_gram)

	file_path = save_dir + "/train_log.txt"
	with open(file_path,"a") as file:
		file.write(mesg+"\n") 


	model_or = great_NET(in_channels,num)
	model = model_or

	# model = nn.DataParallel(model)

	optimizer = Adam(model.parameters(), lr, weight_decay=0.9)
	mse_loss = torch.nn.MSELoss()

	# LOSS network - VGG16
	vgg = Vgg16()
	utils.init_vgg16(vgg, os.path.join(vgg_model_dir, "vgg16.pth"))

	if cuda_flag:
		model.cuda()
		vgg.cuda()
		

	temp_path_model = os.path.join(save_dir)
	temp_path_loss = os.path.join(save_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	
	count = 0
	loss_p1 = 0.
	loss_p2 = 0.
	loss_p3 = 0.
	loss_p4 = 0.

	# retinex loss
	loss_p5 = 0.

	loss_all = 0.


	model.train()

	for e in range(args.epochs):
		img_paths, batch_num = utils.load_dataset(data, batch_size)

		for idx in tqdm(range(batch_num)):

			image_paths_ir = img_paths[idx * batch_size:(idx * batch_size + batch_size)]
			img_ir = utils.get_train_images(image_paths_ir, height=args.Height, width=args.Width, flag=img_flag)

			image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
			img_vi = utils.get_train_images(image_paths_vi, height=args.Height, width=args.Width, flag=img_flag)

			count += 1
			optimizer.zero_grad()
			batch_ir = Variable(img_ir, requires_grad=False)
			batch_vi = Variable(img_vi, requires_grad=False)

			if args.cuda_flag:
				batch_ir = batch_ir.cuda()
				batch_vi = batch_vi.cuda()
			# normalize for each batch
			batch_ir_in = utils.normalize_tensor(batch_ir)
			batch_vi_in = utils.normalize_tensor(batch_vi)

			output = model(batch_ir_in, batch_vi_in)


        
			out_f = output['f']
			out_f = utils.normalize_tensor(out_f)
			out_f = out_f * 255

			out_Lx = output['Lx']
			out_Rx = output['Rx']
			out_Ly = output['Ly']
			out_Ry = output['Ry']
			out_Nx = output['Nx']
			out_Ny = output['Ny']

			
			# pdb.set_trace()

			# ---------- LOSS FUNCTION ----------
			loss_pixel_vi = 10 * mse_loss(out_f, batch_vi)

			# --- Retinex loss ----

			retinex_loss = 1000 * retinex_loss_fuction(out_Lx, out_Rx, out_Ry, out_Nx, batch_ir_in)

			# pdb.set_trace()

			# --- Feature loss ----
			vgg_outs = vgg(out_f)
			vgg_irs = vgg(batch_ir)
			vgg_vis = vgg(batch_vi)

			t_idx = 0
			loss_fea_vi = 0.
			loss_fea_ir = 0.
			loss_gram_ir = 0.
			weights_fea = [lam2_vi, 0.01, 0.5, 0.1]
			weights_gram = [0, 0, 0.1, lam3_gram]

			for fea_out, fea_ir, fea_vi, w_fea, w_gram in zip(vgg_outs, vgg_irs, vgg_vis, weights_fea, weights_gram):
				if t_idx == 0:
					loss_fea_vi = w_fea * mse_loss(fea_out, fea_vi)
				if t_idx == 1 or t_idx == 2:
					# relu2_2, relu3_3, relu4_3
					loss_fea_ir += w_fea * mse_loss(fea_out, w_ir * fea_ir + w_vi * fea_vi)
				if t_idx == 3:
					gram_out = utils.gram_matrix(fea_out)
					gram_ir = utils.gram_matrix(fea_ir)
					loss_gram_ir += w_gram * mse_loss(gram_out, gram_ir)
				t_idx += 1

			total_loss = loss_pixel_vi + loss_fea_vi + loss_fea_ir + loss_gram_ir + retinex_loss
			
			# + retinex_loss

			total_loss.backward()
			optimizer.step()

			loss_p1 += loss_pixel_vi
			loss_p2 += loss_fea_vi
			loss_p3 += loss_fea_ir
			loss_p4 += loss_gram_ir

			loss_p5 += retinex_loss

			loss_all += total_loss

			step = 10
			if count % step == 0:
				loss_p1 /= step
				loss_p2 /= step
				loss_p3 /= step
				loss_p4 /= step
				loss_p5 /= step
				loss_all /= step

				mesg = "{}\t lam2 {}\t w_ir {}\t lam3 {}\t Count {} \t Epoch {}/{} \t Batch {}/{}  \n " \
				       "pixel vi loss: {:.6f} \t fea vi loss: {:.6f} \t " \
				       "fea ir loss: {:.6f}   \t gram ir loss: {:.6f} \n " \
				       "retinex_loss{:.6f}   \t total loss: {:.6f} \n" . \
					format(time.ctime(), lam2_vi, w_ir, lam3_gram, count, e + 1, args.epochs, idx + 1, batch_num,
				           loss_p1, loss_p2, loss_p3, loss_p4, loss_p5,loss_all)
				
				# file_path = save_dir + "/train_log.txt"
				with open(file_path,"a") as file:
					file.write(mesg+"\n") 

				loss_p1 = 0.
				loss_p2 = 0.
				loss_p3 = 0.
				loss_p4 = 0.
				loss_p5 = 0.
				loss_all = 0.

		
		# save model
		model.eval()
		model.cpu()
		save_model_filename = "lrr_net_lam2_" + str(lam2_vi) + "wir_" + str(w_ir) + "_lam3_gram_" + str(lam3_gram) + \
		                      "_epoch_" + str(e + 1) +  ".model"
		save_model_path = os.path.join(temp_path_model, save_model_filename)
		torch.save(model.state_dict(), save_model_path)
		##############
		model.train()
		model.cuda()
		print("\nCheckpoint, trained model saved at: " + save_model_path)

	# save model
	model.eval()
	model.cpu()
	save_model_filename = "fina_" + ".model"
	save_model_path = os.path.join(temp_path_model, save_model_filename)
	torch.save(model.state_dict(), save_model_path) 

	print("\nDone, trained model saved at", save_model_path)

	return save_model_path



def main():
	path = args.path_ir
	train_num = 20000
	data = load_data(path, train_num)

	w_vi = 0.5
	# w_ir = args.w_ir_list[3]
	w_ir = 3.0
	lam2_vi = 2.5
	lam3_gram = 2000

	train(data, lam2_vi, w_ir, w_vi, lam3_gram)

if __name__ == "__main__":
	main()
