import copy
import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree

#TODO select env
from RL.env_binary_question import BinaryRecommendEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from gcn import GraphEncoder
import time
import warnings

from collections import deque

warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    AMAZON: BinaryRecommendEnv,
    YELP_STAR: BinaryRecommendEnv
    }
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature',
    AMAZON: 'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand','prev_act','prev_reward','prev_done','prev_hidden',
                         'next_obvs','next_act','next_reward','next_done','next_hidden'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryPER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, a = 0.6, e = 0.01):
        self.tree =  SumTree(capacity)
        self.capacity = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        
    def push(self, *args):
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        batch_data = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            batch_data.append(data)
            priorities.append(p)
            idxs.append(idx)
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return idxs, batch_data, is_weight
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def __len__(self):
        return self.tree.n_entries

class Multi_Buffer(object):
    def __init__(self,capacity):
        self.capacity=capacity
    def init(self,tasks):
        '''
        	init buffers for all tasks
        '''
        self.task_buffers = dict([(idx, ReplayMemoryPER(capacity=self.capacity))
                                  for idx in tasks])


    def list(self):
        return list(self.task_buffers.keys())

    def push(self, task_id,data):
        '''
            data ==> ('state', 'action', 'next_state', 'reward', 'user_id','next_cand','next_user_id')
        '''
        self.task_buffers[task_id].push(*data)

    def task_size(self, task_id):
        return len(self.task_buffers[task_id])

    def get_buffer(self, task_id):
        return self.task_buffers[task_id]

    def add_buffer(self,task_id):
        self.task_buffers[task_id]=ReplayMemoryPER(capacity=self.capacity)

    def sample(self, task_id, batch_size):
        '''
            Returns tuples of ('state', 'action', 'next_state', 'reward', 'user_id','next_cand','next_user_id') of this task
        '''
        return self.task_buffers[task_id].sample(batch_size)

    def update(self,task_id,idx,errors):
        self.task_buffers[task_id].update(idx,errors)

    def clean(self,task_id):
        self.task_buffers[task_id].clean()

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100,rl2=True,rnn='GRU'):
        super(DQN, self).__init__()
        self.hidden_size=hidden_size
        self.action_size=action_size
        # V(s)
        self.fc2_value = nn.Linear(hidden_size, hidden_size)
        self.out_value = nn.Linear(hidden_size, 1)
        # self.weight_value = nn.Parameter(torch.FloatTensor(1, hidden_size),requires_grad=True)
        #
        # self.bias_value =  nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        #
        # nn.init.kaiming_uniform_(self.weight_value, a=math.sqrt(5))
        # if self.bias_value is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_value)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias_value, -bound, bound)



        # Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)
        self.out_advantage = nn.Linear(hidden_size, 1)
        #
        # self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size),requires_grad=True)
        #
        # self.bias = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.dropout=nn.Dropout(0.5)


        self.rl2 = rl2
        # print('DQN_rl2',self.rl2)
        # self.reset_parameters()
        self.hidden_state=None

        if rnn=='GRU' and rl2:
            self.rnn=nn.GRUCell(hidden_size,hidden_size)
        # else:
        #     self.rnn_val= nn.LSTMCell(hidden_size, hidden_size)
        #     self.rnn_adv=nn.LSTMCell(hidden_size + action_size,hidden_size)


    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         nn.init.uniform_(self.bias, -bound, bound)

    def initial_hidden_state(self,x):

        h_in=torch.zeros((x.shape[0],self.hidden_size),device='cuda')
        return h_in

    def get_last_hidden_state(self,x,h=None,pre_act_rew_done=None):
        if self.rl2:
            if h is None:
                print('h_in initial')
                h_in = self.initial_hidden_state(x)
            else:
                print('h_in_not initial')
                h_in = h
            # combined = self.context(pre_act_rew_done)
            x = torch.cat([x, pre_act_rew_done], dim=-1)
            x = x.view(x.shape[0], x.shape[2])  # N*D
            h_s = self.rnn(x, h_in)  # N*D
            self.hidden_state = h_s
        else:
            return self.hidden_state

        return self.hidden_state

    def forward(self, x, y,h=None,pre_act_rew_done=None,choose_action=True):
        """
        :param x: encode history [N*L*D]; pre_act_rew_done [N * L * D2] ;y: action embedding [N*K*D]
        :return: v: action score [N*K]

        """
        if self.rl2 ==True:
            if h is None:
                print('h_in initial')
                h_in=self.initial_hidden_state(x)
            else:
                print('h_in_not initial')
                h_in = h
            # combined = self.context(pre_act_rew_done)
            x = torch.cat([x, pre_act_rew_done], dim = -1)
            x=x.view(x.shape[0],x.shape[2]) #N*D
            h_s=self.rnn(x,h_in) #N*D

            x=h_s.unsqueeze(1) #N*1*D

            # V(s)
            value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2)  # [N*1*1]

            # Q(s,a)
            if choose_action:
                x = x.repeat(1, y.size(1), 1)
            state_cat_action = torch.cat((x,y),dim=2)
            advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2) #[N*K]


            if choose_action:
                qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
            else:
                qsa = advantage + value

            self.hidden_state=h_s
            return qsa
        else:

            # print('x',x.shape)
            # print('hv_in',hv_in.shape)
            # print('ha_in', ha_in.shape)

            # V(s)
            value = self.out_value(F.relu(self.fc2_value(x))).squeeze(dim=2)  # [N*1*1]
            # value = F.linear(F.relu(self.fc2_value(x)), self.weight_value, self.bias_value).squeeze(dim=2)  # [N*1*1]
            # Q(s,a)
            if choose_action:
                x = x.repeat(1, y.size(1), 1)
            state_cat_action = torch.cat((x, y), dim=2)
            advantage = self.out_advantage(F.relu(self.fc2_advantage(state_cat_action))).squeeze(dim=2) #[N*K]
            # advantage = F.linear(F.relu(self.fc2_advantage(state_cat_action)), self.weight, self.bias).squeeze(
            #     dim=2)  # [N*K]

            if choose_action:
                qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
            else:
                qsa = advantage + value
            self.hidden_state = None
            return qsa


class Agent(object):
    def __init__(self, device, memory, state_size, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID, rl2=True,EPS_START = 0.9, EPS_END = 0.1, EPS_DECAY = 0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.policy_net = DQN(state_size, action_size, hidden_size,rl2).to(device)
        self.target_net = DQN(state_size, action_size, hidden_size,rl2).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(),self.gcn_net.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau
        self.last_hidden_state=None
        self.rl2=rl2


    def select_action(self, state, cand, all_list,action_space, is_test=False, is_last_turn=False):


        pre_obsvs_hist, pre_rewards_hist, pre_actions_hist,pre_done_hist=all_list


        state_emb = self.gcn_net([state])
        cand = torch.LongTensor([cand]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if is_test and (len(action_space[1]) <= 10 or is_last_turn):
            pre_act_rew_done = torch.cat((pre_actions_hist[-1], pre_rewards_hist[-1], pre_done_hist[-1]), dim=-1)
            self.last_hidden_state =self.policy_net.get_last_hidden_state(state_emb,h=self.last_hidden_state,
                                                          pre_act_rew_done=pre_act_rew_done.view(state_emb.shape[0],
                                                                                                 state_emb.shape[1],
                                                                                                 -1))

            return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1],self.last_hidden_state
        with torch.no_grad():
            pre_act_rew_done = torch.cat((pre_actions_hist[-1], pre_rewards_hist[-1], pre_done_hist[-1]), dim=-1)
            print('pre_act_rew_done', pre_act_rew_done.shape)

            actions_value= self.policy_net(state_emb, cand_emb, h=self.last_hidden_state,
                                                          pre_act_rew_done=pre_act_rew_done.view(state_emb.shape[0],
                                                                                                 state_emb.shape[1],
                                                                                                 -1))

            self.last_hidden_state = self.policy_net.hidden_state

            action = cand[0][actions_value.argmax().item()]
            sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]

            return action, sorted_actions.tolist(),self.last_hidden_state


    
    def update_target_model(self):
        #soft assign
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def optimize_model_base(self, BATCH_SIZE, GAMMA):

        if len(self.memory) < BATCH_SIZE:
            return

        self.update_target_model()

        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))
        action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device)  # [N*1]
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.FloatTensor(np.array(batch.reward).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands = []
        for s, c in zip(batch.next_state, batch.next_cand):
            if s is not None:
                n_states.append(s)
                n_cands.append(c)
        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cands)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)

        q_eval = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False)
        # print('qval',q_eval.shape)

        # Double DQN
        best_actions = torch.gather(input=next_cand_batch, dim=1,
                                    index=self.policy_net(next_state_emb_batch, next_cand_emb_batch).argmax(dim=1).view(
                                        len(n_states), 1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)
        q_target = torch.zeros((BATCH_SIZE, 1), device=self.device)
        q_target[non_final_mask] = self.target_net(next_state_emb_batch, best_actions_emb, choose_action=False).detach()
        q_target = reward_batch + GAMMA * q_target

        # prioritized experience replay
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        # mean squared error loss to minimize
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data

    def train(self):
        self.policy_net.train()
        self.gcn_net.train()

    def eval(self):
        self.policy_net.eval()
        self.gcn_net.eval()

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.update_target_model()
        
        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))
        action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device) #[N*1]
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.FloatTensor(np.array(batch.reward).astype(float).reshape(-1, 1)).to(self.device)

        prev_action_batch=torch.cat(batch.prev_act).to(self.device)
        prev_reward_batch = torch.cat(batch.prev_reward).to(self.device)
        prev_done_batch = torch.cat(batch.prev_done).to(self.device)
        #
        # print('prev_action_batch',prev_action_batch.shape)


        prev_action_rew_done=torch.cat((prev_action_batch,prev_reward_batch, prev_done_batch),dim=-1)

        pre_hidden=torch.cat(batch.prev_hidden).to(self.device)


        prev_action_rew_done=prev_action_rew_done.view(state_emb_batch.shape[0],state_emb_batch.shape[1],-1)
        # print('prev_action_batch', prev_action_batch.shape)

        # print('cur_n_hidden_v_batch',cur_n_hidden_v_batch.shape)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands = []
        n_hidden=[]
        n_act=[]
        n_rew=[]
        n_done=[]

        for s, c ,hid,act,rew,don in zip(batch.next_state, batch.next_cand,batch.next_hidden,
                             batch.next_act,batch.next_reward,batch.next_done):
            if s is not None:
                n_states.append(s)
                n_cands.append(c)
                n_hidden.append(hid)

                n_act.append(act)
                n_rew.append(rew)
                n_done.append(don)

        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cands)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)

        next_hidden_batch = torch.cat(n_hidden).to(self.device)


        # print('next_hidden_v_batch', next_hidden_v_batch.shape)



        next_act_batch=torch.cat(n_act).to(self.device)
        next_rew_batch=torch.cat(n_rew).to(self.device)
        next_done_batch=torch.cat(n_done).to(self.device)

        # print('next_act_batch', next_act_batch.shape)


        next_action_rew_done=torch.cat((next_act_batch,next_rew_batch, next_done_batch),dim=-1)


        next_action_rew_done=next_action_rew_done.view(next_state_emb_batch.shape[0],next_state_emb_batch.shape[1],-1)

        q_eval = self.policy_net(state_emb_batch, action_emb_batch,
                                 h=pre_hidden,pre_act_rew_done=prev_action_rew_done,choose_action=False)

        # print('q_eval',q_eval.shape)
        # Double DQN

        best_actions = torch.gather(input=next_cand_batch, dim=1, index=self.policy_net(next_state_emb_batch, next_cand_emb_batch,h=next_hidden_batch,pre_act_rew_done=next_action_rew_done).argmax(dim=1).view(len(n_states),1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)

        q_target = torch.zeros((BATCH_SIZE,1), device=self.device)

        q_target[non_final_mask] =self.target_net(next_state_emb_batch,best_actions_emb,h=next_hidden_batch,
                                                   pre_act_rew_done=next_action_rew_done,choose_action=False).detach()
        q_target = reward_batch + GAMMA * q_target

        # prioritized experience replay
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        # mean squared error loss to minimize
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()

        self.optimizer.zero_grad()

        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data


    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(dataset=data_name, model=self.policy_net, filename=filename+'_'+'policy', epoch_user=epoch_user)
        save_rl_agent(dataset=data_name, model=self.gcn_net, filename=filename + '_' + 'gcn',
                      epoch_user=epoch_user)

        save_rl_agent(dataset=data_name, model=self.optimizer, filename=filename + '_' + 'optim',
                      epoch_user=epoch_user)
        # save_buffer(data_name, self.multi_task_memory, filename+'multi-task-buffer-{}'.format(epoch_user))
        # save_buffer(data_name, self.memory, filename + 'buffer-{}'.format(epoch_user))

    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename+'_'+'policy', epoch_user=epoch_user)
        self.policy_net.load_state_dict(model_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        model_dict = load_rl_agent(dataset=data_name, filename=filename+'_'+'gcn', epoch_user=epoch_user)
        self.gcn_net.load_state_dict(model_dict)

        model_dict = load_rl_agent(dataset=data_name, filename=filename+'_'+'optim', epoch_user=epoch_user)
        self.optimizer.load_state_dict(model_dict)
    
    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)


def train(args, kg, dataset, filename):
    env = EnvDict[args.data_name](kg, dataset, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn, cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                       attr_num=args.attr_num, mode='train', ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)

    memory = ReplayMemoryPER(args.memory_size) #50000
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1,env.ui_embeds.shape[1]))), axis=0))
    gcn_net = GraphEncoder(device=args.device, entity=embed.size(0), emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden).to(args.device)

    if args.rl2 == True:
        reward_dim = 1
        done_dim=1
        input_dim_context =args.hidden +embed.size(1)+ reward_dim + done_dim
        # args.output_dim_conext = (env.action_space.shape[0] + reward_dim) * 2

        # if args.only_concat_context == 3:  # means use LSTM with action_reward_state as an input
        #     input_dim_context =args.hidden_size +64 + reward_dim
        #     actor_idim = [env.observation_space.shape[0] + args.hiddens_conext[0]]
        #     args.output_dim_conext = args.hiddens_conext[0]
        #     dim_others = args.hiddens_conext[0]
        args.hidden=input_dim_context


    agent = Agent(device=args.device, memory=memory, state_size=args.hidden, rl2=args.rl2,action_size=embed.size(1), \
        hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm, PADDING_ID=embed.size(0)-1)

    # agent.load_test_people(people2test,ui_test_valid)
    # self.reward_dict = {
    #     'ask_suc': 0.01,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.1,
    #     'until_T': -0.3,  # until MAX_Turn
    # }
    #ealuation metric  ST@T
    #agent load policy parameters
    if args.load_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.eval_num == 1:
        agent.eval()
        SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
        test_performance.append(SR15_mean)

    for train_step in range(1, args.max_steps+1):
        SR5, SR10, SR15, AvgT, Rank, total_reward = 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        agent.train()
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            if not args.fix_emb:
                state, cand, action_space,acc_fea,support_sets= env.reset(agent.gcn_net.embedding.weight.data.cpu().detach().numpy())  # Reset environment and record the starting state
            else:
                state, cand, action_space,acc_fea,support_sets= env.reset()
            #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
            epi_reward = 0
            is_last_turn = False

            # start of the MDP
            agent.last_hidden_state = None

            enablePrint()  # Enable print function

            # state, cand, action_space, acc_fea, support_sets = env.reset(item=target,user=user)
            ########
            ## create a queue to keep track of past rewards and actions
            ########

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
            # Given batching schema, I need to build a full seq to keep in replay buffer
            # Add to all zeros.
            # zero_action = torch.zeros(64).view(1,-1)
            # zero_obs = {}
            # for _ in range(args.hist_len):
            #     pre_rewards_hist.append(0)
            #     pre_actions_hist.append(copy.deepcopy(zero_action))
            #     pre_obsvs_hist.append(copy.deepcopy(zero_obs))
            #     pre_done_hist.append(0)
            #
            #     # same thing for next_h*
            #     next_hrews.append(0)
            #     next_hacts.append(copy.deepcopy(zero_action))
            #     next_hacts.append(copy.deepcopy(zero_obs))
            #     next_hdone.append(0)

            # now add obs to the seq
            pre_rewards_hist.append(torch.tensor([0.01], device=args.device, dtype=torch.float).view(1,-1)) # reward-1
            pre_actions_hist.append(agent.gcn_net.embedding.weight[acc_fea].view(1,-1).to(args.device))  # action-1
            pre_obsvs_hist.append(state)  # state 0
            pre_done_hist.append(torch.zeros(1,device='cuda').view(1,-1))  # done -1
            pre_hid_hist.append(torch.zeros((1, args.hidden), device='cuda'))
            print('pre_hid_hist',pre_hid_hist[-1].shape)
            for t in count():   # user  dialog
                if t == 14:
                    is_last_turn = True

                action, sorted_actions ,hidden_state= agent.select_action(state, cand,[pre_obsvs_hist,pre_rewards_hist,pre_actions_hist,pre_done_hist],
                                                             action_space, is_last_turn=is_last_turn)


                if not args.fix_emb:
                    next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions, agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                else:
                    next_state, next_cand, action_space, reward, done = env.step(action.item(), sorted_actions)
                #next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)


                ###############
                next_hrews.append(reward.view(1,-1))
                next_hacts.append(agent.gcn_net.embedding.weight[action].view(1, -1))
                next_hobvs.append(next_state)
                next_hdone.append(torch.tensor([done],device=args.device).view(1,-1))
                next_hhid.append(hidden_state)


                if done:
                    next_state = None

                prev_act = pre_actions_hist[-1]
                prev_reward = pre_rewards_hist[-1]
                prev_done = pre_done_hist[-1]
                prev_hidden = pre_hid_hist[-1]
                next_obvs=next_hobvs[-1]

                next_act = next_hacts[-1]
                next_reward = next_hrews[-1]
                next_done = next_hdone[-1]
                next_hidden = next_hhid[-1]


                # # np_next_hacts and np_next_hrews are required for TD3 alg
                # np_next_hacts = np.asarray(next_hacts,
                #                            dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
                # np_next_hrews = np.asarray(next_hrews, dtype=np.float32)  # (hist, )
                # np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()  # (hist, )

                agent.memory.push(state, action, next_state, reward, next_cand,
                                  prev_act,
                                  prev_reward,
                                  prev_done,
                                  prev_hidden,
                                  next_obvs,
                                  next_act,
                                  next_reward,
                                  next_done,
                                  next_hidden)


                # new becomes old
                if next_state is not None:
                    pre_rewards_hist.append(reward.view(1,-1))  # reward t [1]
                    pre_actions_hist.append(agent.gcn_net.embedding.weight[action].view(1, -1))  # action t [1,D]
                    # pre_obsvs_hist.append(obs.copy()) #???
                    pre_obsvs_hist.append(next_state)  # state t+1 [1,D]
                    pre_done_hist.append(torch.tensor([done],device=args.device).view(1,-1))  # done t
                    pre_hid_hist.append(hidden_state)


                state = next_state
                cand = next_cand

                if args.rl2:
                    print('Meta Calculation')
                    newloss = agent.optimize_model(args.batch_size, args.gamma)

                else:
                    print('UNICORN base calculation')
                    newloss= agent.optimize_model_base(args.batch_size, args.gamma)

                if newloss is not None:
                    loss += newloss

                if done:
                    # every episode update the target model to be same with model
                    #agent.update_target_model()
                    if reward.item() == 1:  # recommend successfully
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
                    AvgT += t+1
                    total_reward += epi_reward
                    break
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, rewards:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.sample_times, SR10 / args.sample_times, SR15 / args.sample_times,
                                                AvgT / args.sample_times, Rank / args.sample_times, total_reward / args.sample_times, args.sample_times))

        PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + 'train_log' + '.txt'
        with open(PATH, 'a') as f:
            f.write('Training epocch:{}\n'.format(train_step*args.sample_times))
            f.write('===========Test Turn===============\n')
            f.write('Training {} user tuples\n'.format(args.sample_times))
            f.write('Training SR-turn@{}: {}\n'.format(15, SR15 / args.sample_times))
            f.write('Training avg-turn: {}\n'.format(AvgT / args.sample_times))
            f.write('================================\n')

        #if train_step % args.eval_num == 0 or (train_step < 1000 and train_step % (args.eval_num/10) == 0):
        if train_step % args.eval_num == 0:
            agent.eval()
            SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, train_step)
            test_performance.append(SR15_mean)

        if train_step % args.save_num == 0:
            agent.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    print(test_performance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR,AMAZON],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method', type=str, default='weight_entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    #parser.add_argument('--update_times', type=int, default=100, help='the epoch of updating')
    parser.add_argument('--max_steps', type=int, default=100, help='max training steps')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')
    parser.add_argument('--observe_num', type=int, default=100, help='candidate item sampling number')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default='transe', help='pretrained embeddings')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    parser.add_argument('--rl2', action='store_false', help='use rl2 or not')
    parser.add_argument('--hist_len', type=int,default=1, help='previous length')
    parser.add_argument('--method', type=str, default='base', help='use method')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('-rl2',args.rl2)
    print('args.entropy_method:', args.entropy_method)

    dataset = load_dataset(args.data_name)
    filename = '{}-train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(args.method,
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    train(args, kg, dataset, filename)

if __name__ == '__main__':
    main()