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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def extract_first_14_digits(input_string):
    # 定义匹配14个数字的正则表达式
    pattern = r'\d{14}'
    # 使用正则表达式进行匹配，只获取第一个匹配结果
    match = re.search(pattern, input_string)
    if match:
        return match.group()
    else:
        return None

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

def run(infrared_path,visible_path,model,img_flag,output_dir,img_name):
    img_ir = utils.get_train_images(infrared_path, height=None, width=None, flag=img_flag)
    img_vi = utils.get_train_images(visible_path, height=None, width=None, flag=img_flag)

    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)
    if args.cuda_flag:
        img_ir = img_ir.cuda()
        img_vi = img_vi.cuda()

    img_ir = utils.normalize_tensor(img_ir)
    img_vi = utils.normalize_tensor(img_vi)

    output = model(img_ir, img_vi)

    out = output['f']
    out_path = output_dir + img_name + '.png'


    utils.save_image(out, out_path)

def fusion(model_path):
    test_dir = args.test_dir
    imgs_paths_ir, names = utils.list_images(test_dir)
    num = len(imgs_paths_ir)
   

    if args.img_flag == True:
        in_channels = 3
    else:
        in_channels = 1

    output_path_root = args.output_path_root

    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)


    path = extract_first_14_digits(model_path)

    # output_path = output_path_root + '/' + path + '_epoch4' + '/'
    output_path = output_path_root + '/' + path  + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with torch.no_grad():
        model = load_model(model_path,in_channels=in_channels,num=args.num)
        for i in tqdm(range(num)):
            img_name = names[i]
            infrared_path = imgs_paths_ir[i]
            visible_path = imgs_paths_ir[i].replace('lwir', 'visible')
            if visible_path.__contains__('IR'):
                visible_path = visible_path.replace('IR', 'VIS')
            else:
                visible_path = visible_path.replace('i.', 'v.')
                
            run(infrared_path,visible_path,model,args.img_flag,output_path,img_name)



def main():
    model_path = "/home/mchen/model/NewNet//20240403112949/fina_.model"
    fusion(model_path)

if __name__ == '__main__':
	main()
