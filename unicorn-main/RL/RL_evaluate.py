import time
import argparse
from itertools import count
import torch.nn as nn
import torch
import math
from collections import namedtuple
from utils import *
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
from tqdm import tqdm
EnvDict = {
        LAST_FM: BinaryRecommendEnv,
        LAST_FM_STAR: BinaryRecommendEnv,
        YELP: EnumeratedRecommendEnv,
        YELP_STAR: BinaryRecommendEnv,
    AMAZON:BinaryRecommendEnv,
    }
from collections import deque
def dqn_evaluate(args, kg, dataset, agent, filename, i_episode):
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num, mode='test', ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    tt = time.time()
    start = tt
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1
    # }
    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    plot_filename = 'Evaluate-'.format(i_episode) + filename
    if args.data_name in [LAST_FM_STAR, LAST_FM,AMAZON]:
        if args.eval_num == 1:
            test_size = 500
        else:
            test_size = 2000     # Only do 2000 iteration for the sake of time
        user_size = test_size
    if args.data_name in [YELP_STAR, YELP]:
        if args.eval_num == 1:
            test_size = 500
        else:
            test_size = 2500     # Only do 2500 iteration for the sake of time
        user_size = test_size

    # test_size = 500
    # user_size = 500
    # print('The select Test size : ', test_size)



    for user_num in tqdm(range(user_size)):  #user_size
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))
        #state, cand, action_space = test_env.reset()  # Reset environment and record the starting state
        if not args.fix_emb:
            state, cand, action_space,acc_fea,_= test_env.reset(agent.gcn_net.embedding.weight.data.cpu().detach().numpy())  # Reset environment and record the starting state
        else:
            # user = test_array[user_num][0], item = test_array[user_num][1]
            state, cand, action_space,acc_fea,_= test_env.reset()
        agent.last_hidden_state = None

        pre_rewards_hist = deque(maxlen=args.hist_len)
        pre_actions_hist = deque(maxlen=args.hist_len)
        pre_obsvs_hist = deque(maxlen=args.hist_len)
        pre_done_hist = deque(maxlen=args.hist_len)
        pre_hid_hist = deque(maxlen=args.hist_len)

        next_hrews = deque(maxlen=args.hist_len)
        next_hacts = deque(maxlen=args.hist_len)
        next_hobvs = deque(maxlen=args.hist_len)
        next_hdone = deque(maxlen=args.hist_len)
        next_hhid = deque(maxlen=args.hist_len)

        pre_rewards_hist.append(torch.tensor([0.01], device=args.device, dtype=torch.float).view(1, -1))  # reward-1
        pre_actions_hist.append(agent.gcn_net.embedding.weight[acc_fea].view(1, -1).to(args.device))  # action-1
        pre_obsvs_hist.append(state)  # state 0
        pre_done_hist.append(torch.zeros(1, device='cuda').view(1, -1))  # done -1
        pre_hid_hist.append(torch.zeros((1, args.hidden), device='cuda'))
        print('pre_hid_hist', pre_hid_hist[-1].shape)

        #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
        epi_reward = 0
        is_last_turn = False
        for t in count():  # user  dialog
            if t == 14:
                is_last_turn = True
            action, sorted_actions,hidden_state= agent.select_action(state, cand, [pre_obsvs_hist,pre_rewards_hist,pre_actions_hist,pre_done_hist],action_space,
                                                                     is_test=True, is_last_turn=is_last_turn)



            next_state, next_cand, action_space, reward, done = test_env.step(action.item(), sorted_actions)
            #next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            epi_reward += reward
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)

            next_hrews.append(reward.view(1, -1))
            next_hacts.append(agent.gcn_net.embedding.weight[action].view(1, -1))
            next_hobvs.append(next_state)
            next_hdone.append(torch.tensor([done], device=args.device).view(1, -1))
            next_hhid.append(hidden_state)

            # new becomes old
            pre_rewards_hist.append(reward.view(1, -1))  # reward t [1]
            pre_actions_hist.append(agent.gcn_net.embedding.weight[action].view(1, -1))  # action t [1,D]

            pre_obsvs_hist.append(next_state)  # state t+1 [1,D]
            pre_done_hist.append(torch.tensor([done], device=args.device).view(1, -1))  # done t
            pre_hid_hist.append(hidden_state)


            if done:
                next_state = None
            state = next_state
            cand = next_cand
            if done:
                enablePrint()
                if reward.item() == 1:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1
                    Rank += (1/math.log(t+3,2) + (1/math.log(t+2,2)-1/math.log(t+3,2))/math.log(done+1,2))
                else:
                    Rank += 0
                total_reward += epi_reward
                AvgT += t+1
                break
        
        if (user_num+1) % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num, Rank / args.observe_num, total_reward / args.observe_num]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, Rank / args.observe_num, total_reward / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
        enablePrint()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    reward_mean = np.mean(np.array([item[5] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{}'.format(SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + plot_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(i_episode, SR15_mean, AvgT_mean, Rank_mean, reward_mean))

    return SR15_mean


