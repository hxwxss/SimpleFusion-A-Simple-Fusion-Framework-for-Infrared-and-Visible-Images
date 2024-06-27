class Args():
	# For training
	path_ir = '/home/mchen/dataset_KAIST/KAIST/lwir'
	
	#  img_flag = true ---> RGB, false ---> gray
	img_flag = False

	cuda_flag = True
	lr = 0.00001
	epochs = 4
	batch_size = 8

	vgg_model_dir = '/home/mchen/model/VGG16/'


# 	# # Network Parameters
	Height = 128
	Width = 128

	w_vi = 0.5
	w_ir_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	lam2_vi_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
	lam3_gram_list = [1500, 2000, 2500]

	num = 128 



	save_dir = "/home/mchen/model/NewNet/"

	# For testing
	test_dir = "/home/mchen/input_for_test/40_pairs_tno_vot_new/lwir"
	model_path = "/home/mchen/model/NewNet//20240321105153/fina_.model"
	output_path_root = "/home/mchen/output_for_test"







