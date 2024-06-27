import os
import torch
import numpy as np
from torch.autograd import Variable
from net import great_NET, Vgg16
from args import Args as args
import utils
from tqdm import tqdm
from datetime import datetime
# import pdb
import re



def load_model(path,in_channels,num):
	
	model = great_NET(in_channels,num)
	model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
	
	total = sum([param.nelement() for param in model.parameters()])
	print('Number	of	parameter: {:4f}M'.format(total / 1e6))
	
	model.eval()
	model.cuda()

	return model

path = "/home/mchen/model/NewNet(low+high)/20240321140042/fina_.model"
model = load_model(path,1,args.num)