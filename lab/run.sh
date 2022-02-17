#/bin/bash

srun --time 3-0 --qos=32gpu7d --gres=gpu:1 python run_lab.py /home/rm360179/samsung/image-colorization_d6a566/landscape_images \
                                                  --gpu \
						  --alpha 0.01 \
						  --loss_type="bright1" \
						  --lr 5e-4 \
						  #--test_run \
