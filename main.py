from args import Args as args
from train import load_data
from train import train
import os
from test_fusion import fusion

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    
    w_ir_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    lam2_vi_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    lam3_gram_list = [1500, 2000, 2500]

    for w_ir in w_ir_list[0:3]:
        for lam2_vi in lam2_vi_list:
            for lam3_gram in lam3_gram_list:
                
                path = args.path_ir
                train_num = 20000
                data = load_data(path, train_num)

                w_vi = 0.5
                model_path = train(data, lam2_vi, w_ir, w_vi, lam3_gram)
                fusion(model_path)



if __name__ == '__main__':

    main()
