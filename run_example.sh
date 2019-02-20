#Train 
python3 train.py --model NL34_LinkNet --name 'NL34_LinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load "" 
#Train w loading (download or train models before run this line) 
python3 train.py --model NL34_LinkNet --name 'NL34_LinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load "weights/NL34_LinkNet"

#Train different locations 
python3 train.py --model Baseline --name 'Baseline' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model NL3_LinkNet --name 'NL3_LinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model NL4_LinkNet --name 'NL4_LinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model NL34_LinkNet --name 'NL34_LinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""

#Train different pairwise functions 
python3 train.py --model Baseline --name 'Baseline' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model NL_LinkNet_DotProduct --name 'NL_LinkNet_DotProduct' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model NL_LinkNet_Gaussian --name 'NL_LinkNet_Gaussian' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model NL_LinkNet_EGaussian --name 'NL_LinkNet_EGaussian' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""

#Train benchmarks 
python3 train.py --model DLinkNet --name 'DLinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model LinkNet --name 'LinkNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""
python3 train.py --model UNet --name 'UNet' --crop_size 1024 1024 --init_lr 0.0003 --dataset '../dataset/Road/train/' --load ""


#Test 
python3 test.py --model model_name --name 'weight name' --source '/path/of/source/' --scales 1.0 --target 'submit dir name'

#Test on valid set 
python3 test.py --model NL34_LinkNet --name 'NL34_LinkNet' --source '../dataset/Road/valid' --scales 1.0 --target 'NL4_LinkNet_valid'

#Test on test set
python3 test.py --model NL34_LinkNet --name 'NL34_LinkNet' --source '../dataset/Road/test' --scales 1.0 --target 'NL4_LinkNet_test'

#multi scale testing in series. 
model=NL34_LinkNet 
weight_name='NL34_LinkNet_ReRe' 

python3 test.py --model $model --name $weight_name --scales 0.75 1.0 1.25 --target 'NL34_LinkNet_MST0.75_1.25'
python3 test.py --model $model --name $weight_name --scales 1.0 1.125 --target 'NL34_LinkNet_MST1.25'
#python3 test.py --model $model --name $weight_name --scales 1.0 1.5 --target 'NL34_LinkNet_MST1.5'
#python3 test.py --model $model --name $weight_name --scales 1.0 1.125 1.25 --target 'NL34_LinkNet_MST1.125_1.25'
#python3 test.py --model $model --name $weight_name --scales 1.0 1.25 1.5  --target 'NL34_LinkNet_MST1.25_1.5'
#python3 test.py --model $model --name $weight_name --scales 1.0 1.125 --target 'NL34_LinkNet_MST1.125'



