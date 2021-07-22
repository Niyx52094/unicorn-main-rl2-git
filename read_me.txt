cd D:\CS\NTU\MSC_PROJ\RL_code\unicorn-main-rl2-git\unicorn-main

//python graph_init.py --data_name LAST_FM 

//python FM_train.py --data_name LAST_FM

直接调用一下代码
Meta rl2:

## LAST_FM

python RL_model.py --data_name LAST_FM --fm_epoch 0 --eval_num 1 --method meta --save_num 1 

## AMAZON

python RL_model.py --data_name AMAZON --fm_epoch 0 --eval_num 1 --method meta --save_num 1


Base：

## LAST_FM

python RL_model.py --data_name LAST_FM --fm_epoch 0 --eval_num 1 --method base --save_num 1 --rl2

## AMAZON

python RL_model.py --data_name AMAZON --fm_epoch 0 --eval_num 1 --method base --save_num 1 --rl2