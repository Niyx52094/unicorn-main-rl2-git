# unicorn-main-rl2-git

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
