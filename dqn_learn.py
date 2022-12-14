from distutils import core
from importlib.resources import path
from platform import node
from queue import Empty
from turtle import shape
from typing import List

import math

from requests import request

import network as nt
import config as cf
import content as ct
import scenario as sc

from gym.spaces import Discrete, Box
from replaybuffer import ReplayBuffer
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import copy

import savingfunctions as sf
# Qnetwork
class DQN(Model):

    def __init__(self, action_n, state_dim):
        super(DQN, self).__init__()

        self.h1 = Dense(cf.H1 * state_dim, activation='relu')
        self.d1 = Dropout(rate = cf.DROPOUT_RATE)
        self.h2 = Dense(cf.H2 * state_dim, activation='relu')
        self.d2 = Dropout(rate = cf.DROPOUT_RATE)
        self.h3 = Dense(cf.H3 * state_dim, activation='relu')
        self.d3 = Dropout(rate = cf.DROPOUT_RATE)
        self.h4 = Dense(cf.H4 * state_dim, activation='relu')
        self.d4 = Dropout(rate = cf.DROPOUT_RATE)
        self.h5 = Dense(cf.H5 * state_dim, activation='relu')
        self.d5 = Dropout(rate = cf.DROPOUT_RATE)
        self.h6 = Dense(cf.H6 * state_dim, activation='relu')
        self.d6 = Dropout(rate = cf.DROPOUT_RATE)
        self.h7 = Dense(cf.H7 * state_dim, activation='relu')
        self.d7 = Dropout(rate = cf.DROPOUT_RATE)
        self.h8 = Dense(cf.H8 * state_dim, activation='relu')
        self.d8 = Dropout(rate = cf.DROPOUT_RATE)
        self.q = Dense(action_n, activation='linear')

    def call(self, x):
        x = self.h1(x)
        x = self.d1(x)
        x = self.h2(x)
        x = self.d2(x)
        x = self.h3(x)
        x = self.d3(x)
        x = self.h4(x)
        x = self.d4(x)
        x = self.h5(x)
        x = self.d5(x)
        x = self.h6(x)
        x = self.d6(x)
        x = self.h7(x)
        x = self.d7(x)
        x = self.h8(x)
        x = self.d8(x)
        q = self.q(x)
        return q


class DQNagent():

    def __init__(self):

        self.network = nt.Network()
        self.twin_network:nt.Network
        # state ??????
        # 1. DataCenter ?????? ?????? ????????? ??????
        # 2. BS ?????? ?????? ????????? ??????
        # 3. MicroBS ?????? ?????? ????????? ??????

        # 4???????????? ?????????
        # 4. ???????????? ?????? ??????

# !       self.DataCenter_AR = self.get_AR("DataCenter")
# !       self.BS_AR = self.get_AR("BS")
# !       self.MicroBS_AR = self.get_AR("MicroBS")

        # BackBone ??? Data Center ?????? ??? ????????? ???????
        # [path ??? Mirco Base Station??? ??????, path ??? Base Station??? ??????,DataCenter??? ??????, ???????????? ?????? ?????????]
        self.action_space = Discrete(4)
        self.observation_space = Box(-1,1,shape=(3,))
        self.action_n = 4
        # path??? [node, Micro BS, BS, Data center, Core Internet]
        self.path = []

        # state ???????????? ??????, ???????????? ?????? ??????, ?????? ?????? ?????? ??????
        # ! ????????? BS??? ?????? ?????? ????????? ?????????
        # ! ???????????? ???????????? ????????????
        # ! ???????????? ??????(Round%7)
        # ! ?????? : ???  BS??? ???????????? , 1. ????????? ???????????? 2. ???????????? ??????
        self.round_nb = 0
        self.round_day = 0
        self.state:np.array = self.set_state()
        #print("init ???????????? self.state")
        #print(self.state)
        self.state_dim = self.state.shape[0]
        
    
        # DQN ?????????????????????
        self.GAMMA = cf.GAMMA
        self.BATCH_SIZE = cf.BATCH_SIZE
        self.BUFFER_SIZE = cf.BUFFER_SIZE
        self.DQN_LEARNING_RATE = cf.DQN_LEARNING_RATE
        self.TAU = cf.TAU
        self.EPSILON = cf.EPSILON
        self.EPSILON_DECAY = cf.EPSILON_DECAY
        self.EPSILON_MIN = cf.EPSILON_MIN
        
        ## create Q networks
        self.dqn = DQN(self.action_n, self.state_dim)
        self.target_dqn = DQN(self.action_n, self.state_dim)

        self.dqn.build(input_shape=(None, self.state_dim))
        self.target_dqn.build(input_shape=(None, self.state_dim))

        self.dqn.summary()

        # optimizer
        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []
        self.save_epi_cache_hit_rate = []
        self.save_epi_redundancy = []
        self.save_epi_avg_hop = []
        self.save_epi_existing_content = []
        self.save_epi_denominator = []

        # ????????? ??? reward front/back
        self.save_epi_round_reward_front = []
        self.save_epi_round_reward_back = []
        self.save_epi_round_reward_d_core = []
        self.save_epi_round_reward_d_cache = []
        self.save_epi_round_reward_c_node = []
        self.save_epi_round_reward_alpha_redundancy = []
        self.save_epi_round_reward_beta_redundancy = []
        self.save_epi_round_reward_vacancy = []

        # ADAM
        self.steps = 0
        self.memory = deque(maxlen = 10000)

        # reward parameter
        self.a = cf.a
        self.b = cf.b
        self.c = cf.c
        self.d = cf.d
        self.e = cf.e

        self.d_core = 0
        self.d_cache = 0
        self.R_cache = 0
        self.H_arg = 0
        self.c_node = 0
        self.stored_type = 0
        self.stored_nodeID = 0
        self.alpha_redundancy = 0
        self.beta_redundancy = 0
        self.vacancy = 0

        # Done ?????? action??? 7000??? ???????????? ???
        self.NB_ACTION = cf.NB_ACTION
        self.stop = self.NB_ACTION
        self.action_cnt = 0
        self.step_cnt = 0
        
        # cache hit count ==> network.py??? ?????????????????? ??????
        self.cache_hit_cnt = 0
        self.hop_cnt = 0

        # saving function
        self.sf = sf.savingfunctions()
        
        # request dictionary
        self.requestDictionary = self.set_requestDictionary()
        self.contentLabelDictionary = self.set_contentLabelDictionary()
        self.actionDictionary = self.set_actionDictionary()
        self.roundRequestDictionary = self.set_roundRequestDictionary()
        
        # cache ?????? Flag [0,1]
        # 0 : ?????? ??????
        # 1 : ?????? ???
        self.cache_changed = 0
        self.cache_changed_cnt = 0
        self.action_3_cnt = 0
        self.date = 0

        self.gamma_episode_reward = 0
        self.set_qs_action_count_list()
        self.set_random_action_count_list()

        

    def reset(self):
        self.network = nt.Network()

        # state ?????? ?????? round_day??? ????????????
        self.state = self.set_state()

        self.save_epi_round_reward_front = []
        self.save_epi_round_reward_back = []
        self.save_epi_round_reward_d_core = []
        self.save_epi_round_reward_d_cache = []
        self.save_epi_round_reward_c_node = []
        self.save_epi_round_reward_alpha_redundancy = []
        self.save_epi_round_reward_beta_redundancy = []
        self.save_epi_round_reward_vacancy = []

        return self.state

    def set_tmp_state(self):
        
        # state = [
        #   round_day,
        #   requested_content, request count of requested_content, path
        #   MicroBS: (stored content label, request count of stored content label), ... ,(stored content label, request count of stored content label)
        #   .
        #   .
        #   BS: (stored content label, request count of stored content label), ... ,(stored content label, request count of stored content label)
        #   .
        #   .
        #   DataCenter: (stored content label, request count of stored content label), ... ,(stored content label, request count of stored content label)
        # ]
        state = []

        round_day =  self.network.days[self.round_nb] % 7
        self.current_requested_content, self.current_path = self.network.request_and_get_path(round_day)
        self.current_full_path = self.network.get_simple_path(self.current_path[0])

        title = self.current_requested_content.get_title()
        self.update_requestDictionary(title)
        self.updateRoundRequestDictionary(title, round_day)
        state.append(round_day)
        state.append(self.contentLabelDictionary[title])
        state.append(self.requestDictionary[title])
        
        for i in range(1,4):
            state.append(self.current_full_path[i])

        # MicroBS [cache available, Labels of contents in storage]
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            content_cnt = len(self.network.microBSList[i].storage.storage)
            for j in range(content_cnt):
                title = self.network.microBSList[i].storage.storage[j].get_title()
                content_label = self.contentLabelDictionary[title]
                content_request_count = self.requestDictionary[title]
                state.append(content_label)
                state.append(content_request_count)
            
            # storage ??? ??? ????????? ?????? ???, ???????????? ????????? label = 0 ??? ?????????
            #print(content_cnt)
            while content_cnt != cf.microBS_SIZE/cf.CONTENT_SIZE:
                content_cnt = content_cnt + 1
                # content label
                state.append(0)
                # content request count
                state.append(0)
                #print("0 ?????????")

        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            content_cnt = len(self.network.BSList[i].storage.storage)
            for j in range(content_cnt):
                title = self.network.BSList[i].storage.storage[j].get_title()
                content_label = self.contentLabelDictionary[title]
                content_request_count = self.requestDictionary[title]
                state.append(content_label)
                state.append(content_request_count)
            #print(content_cnt)
            # storage ??? ??? ????????? ?????? ???, ???????????? ????????? label = 0 ??? ?????????
            while content_cnt != cf.BS_SIZE/cf.CONTENT_SIZE:
                content_cnt = content_cnt + 1
                # content label
                state.append(0)
                # content request count
                state.append(0)
                #print("0 ?????????")

        # DataCenter
        content_cnt = len(self.network.dataCenter.storage.storage)
        for i in range(content_cnt):
            title = self.network.dataCenter.storage.storage[i].get_title()
            content_label = self.contentLabelDictionary[title]
            content_request_count = self.requestDictionary[title]
            state.append(content_label)
            state.append(content_request_count)

        #print(content_cnt)
        # storage ??? ??? ????????? ?????? ???, ???????????? ????????? label = 0 ??? ?????????
        while content_cnt != cf.CENTER_SIZE/cf.CONTENT_SIZE:
            content_cnt = content_cnt + 1
            # content label
            state.append(0)
            # content request count
            state.append(0)
            #print("0 ?????????")

        state = np.array(state)
        #print(state)
        return state
    
    def set_state(self):
        
        # TODO : round_day, requested content, Base Station ?????? ??????
        # ?????? ????????? ????????? DNN ??? ?????? ??? 

        state = []

        # network.py ?????? round_day ??? state??? ????????? ??????
        # ????????? step ?????? ?????? ????????????
        #print("set_state ?????? round_nb: {}".format(self.round_nb))
        round_day =  self.network.days[self.round_nb] % 7
        state.append(round_day)
        

        
        # MicroBS [cache available, Labels of contents in storage]
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            state.append(cf.microBS_SIZE - self.network.microBSList[i].storage.stored)
            
            content_cnt = len(self.network.microBSList[i].storage.storage)
            for j in range(content_cnt):
                title = self.network.microBSList[i].storage.storage[j].get_title()
                content_label = self.contentLabelDictionary[title]
                state.append(content_label)
            
            # storage ??? ??? ????????? ?????? ???, ???????????? ????????? label = 0 ??? ?????????
            #print(content_cnt)
            while content_cnt != cf.microBS_SIZE/cf.CONTENT_SIZE:
                content_cnt = content_cnt + 1
                state.append(0)
                #print("0 ?????????")

        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            state.append(cf.BS_SIZE - self.network.BSList[i].storage.stored)
            
            content_cnt = len(self.network.BSList[i].storage.storage)
            for j in range(content_cnt):
                title = self.network.BSList[i].storage.storage[j].get_title()
                content_label = self.contentLabelDictionary[title]
                state.append(content_label)
            #print(content_cnt)
            # storage ??? ??? ????????? ?????? ???, ???????????? ????????? label = 0 ??? ?????????
            while content_cnt != cf.BS_SIZE/cf.CONTENT_SIZE:
                content_cnt = content_cnt + 1
                state.append(0)
                #print("0 ?????????")

        # DataCenter
        state.append(cf.CENTER_SIZE - self.network.dataCenter.storage.stored)
        content_cnt = len(self.network.dataCenter.storage.storage)
        for i in range(content_cnt):
            title = self.network.dataCenter.storage.storage[i].get_title()
            content_label = self.contentLabelDictionary[title]
            state.append(content_label)

        #print(content_cnt)
        # storage ??? ??? ????????? ?????? ???, ???????????? ????????? label = 0 ??? ?????????
        while content_cnt != cf.CENTER_SIZE/cf.CONTENT_SIZE:
            content_cnt = content_cnt + 1
            state.append(0)
            #print("0 ?????????")

        state = np.array(state)

        return state


        
    def memorize(self, state, action, reward, next_state, done):

        self.memory.append(state,action,reward,next_state, done)

    def choose_action(self, state):
        
        #print("choose_action ?????????")
        if np.random.random() <= self.EPSILON:
            action = self.action_space.sample()
            #print("random action : {}".format(action))
            self.random_action_count_list[action] += 1
            return action
        else:
            qs = self.dqn(tf.convert_to_tensor([state],dtype=tf.float32))
            #if self.round_nb % 1000 == 0:
            #    print("qs : {} \t max : {}".format(qs, np.argmax(qs.numpy())))
            #print("np.argmax(qs.numpy()) : {}".format(np.argmax(qs.numpy())))
            self.qs_action_count_list[np.argmax(qs.numpy())] += 1
            return np.argmax(qs.numpy())
        
    def update_target_network(self, TAU):
        phi = self.dqn.get_weights()
        #print("phi : {}".format(phi))
        target_phi = self.target_dqn.get_weights()

        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_dqn.set_weights(target_phi)

    def dqn_learn(self, state, actions, td_targets):
        #print("dqn learn ?????????")
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_n)
            q = self.dqn(state, training=True)
            #print(q)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)    # ?????? ?????? ??? ?????? ???
            loss = tf.reduce_mean(tf.square(q_values-td_targets))                   # MSE 
            
        #print("dqn ?????? ??????")
        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.dqn_opt.apply_gradients(zip(grads, self.dqn.trainable_variables))

    def td_target(self, rewards, target_qs, dones):
        #print("td_target ??????")
        max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * max_q[i]

        #print("td_target ??????")
        ##print(y_k)
        return y_k
    
    ## load actor weights
    def load_weights(self, path):
        self.dqn.load_weights(path + 'networkSIM_dqn.h5')

    def step(self, action, path, requested_content):
        #print("step function ?????? ?????????")
        self.step_cnt = self.step_cnt + 1
        # ?????? ????????? ?????? ?????? {??????, ????????? ?????????}
        self.action_cnt = self.action_cnt + 1
        self.stop = self.stop - 1
        self.act(path, requested_content, action)

        #! ?????? ?????? ?????? ??????????????? ??????
        # @ round_day ??? state??? ???????????? ????????? 
        # @ next_state ??? ????????? ?????? ?????????
        if self.stop != 0:
            self.round_nb += 1
            done = False
        else:
            done = True
            self.stop = self.NB_ACTION
            self.last_round_nb = self.round_nb + 1
            self.round_nb = 0


        next_state = self.set_state()
        reward = self.get_reward(action, path, requested_content)

        return next_state, reward, done

    def reset_parameters(self):

        self.round_nb, self.cache_hit_cnt, self.action_cnt, self.step_cnt, self.hop_cnt = 0,0,0,0,0

        # DELETE PERIOD ??? ????????? Storage update
        self.date = 0
        self.lastdate = 0

        self.gamma_episode_reward = 0

        self.cache_changed_cnt = 0
        self.action_3_cnt = 0

        self.requestDictionary = self.set_requestDictionary()
        self.actionDictionary = self.set_actionDictionary()
        self.roundRequestDictionary = self.set_roundRequestDictionary()
        self.set_qs_action_count_list()
        self.set_random_action_count_list()

    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):
            print("\n\n{} EPSILON : {}".format(ep+1, self.EPSILON))
            print("buffer count : {}".format(self.buffer.buffer_count()))
            # reset episode
            time, episode_reward, done = 0, 0, False
            
            self.reset_parameters()

            # reset the environment and observe the first state
            #print("reset?")
            state = self.reset()

            #print("reset ?????? state : {}".format(state))
            while not done:

                # @ round_day ??? state ??? ????????? 
                # @ reset ??? ??? ??????
                self.date = self.network.days[self.round_nb]
                round_day =  self.date % 7
                #print(self.round_nb)
                requested_content, path = self.network.request_and_get_path(round_day)
                title = requested_content.get_title()

                self.update_requestDictionary(title)
                self.updateRoundRequestDictionary(title, round_day)

                if self.date != self.lastdate:
                    self.deleteOutofdateContents()
                # ?????? 
                self.hop_cnt += len(path) - 1
                self.lastdate = self.date

                # ????????? ?????? && ?????? ?????????????????? cache hit ??? ???????????????
                if len(path) >= 4:
                    
                    if len(path) == 4:
                        ct.updatequeue(path,requested_content,self.network.microBSList,self.network.BSList,self.network.dataCenter, self.date)
                        self.cache_hit_cnt += 1
                        

                    # pick an action
                    action = self.choose_action(state)

                    # @ actionDictionary ??? action append ??????
                    self.append_actionDictionary(title, action)

                    #print("choose_action ???")
                    # observe reward, new_state
                    #print("state : {}".format(state))
                    next_state, reward, done = self.step(action, path, requested_content)
                    #print("next_state : {}".format(next_state))
                    train_reward = reward

                    # add transition to replay buffer
                    self.buffer.add_buffer(state, action, train_reward, next_state, done)

                    if self.buffer.buffer_count() > cf.BUFFER_SIZE/2:  # start train after buffer has some amounts

                        # decaying EPSILON
                        if self.EPSILON > self.EPSILON_MIN:
                            self.EPSILON *= self.EPSILON_DECAY
                        
                        # sample transitions from replay buffer
                        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                        
                        # predict target Q-values
                        target_qs = self.target_dqn(tf.convert_to_tensor(next_states, dtype=tf.float32))

                        # compute TD targets
                        y_i = self.td_target(rewards, target_qs.numpy(), dones)

                        self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32), actions, tf.convert_to_tensor(y_i, dtype=tf.float32))
                        
                        # update target network
                        self.update_target_network(self.TAU)
                    

                # MicroBS, BS ?????? cache hit ??? ???????????? ???
                else:
                    self.cache_hit_cnt += 1
                    self.round_nb += 1
                    ct.updatequeue(path,requested_content,self.network.microBSList,self.network.BSList,self.network.dataCenter, self.date)
                    next_state = self.set_state()

                # update current state
                state = next_state
                episode_reward += reward

                self.gamma_episode_reward = self.GAMMA*self.gamma_episode_reward + reward
                time += 1
                #print(time)

            ## display rewards every episode
            
            cache_hit_rate = self.cache_hit_cnt/time

            # action 0,1,2 ??? ???????????? cache changed ratio ??????
            self.cache_changed_ratio = self.cache_changed_cnt/(self.action_cnt-self.action_3_cnt)


            redundancy, existing_content, denominator = self.function1()
            avg_hop = self.hop_cnt/time
            qs_action_cnt = sum(self.qs_action_count_list)
            random_action_cnt = sum(self.random_action_count_list)
            #print(self.actionDictionary)
            self.write_actionDictionary_file(ep+1, self.requestDictionary, self.actionDictionary)

            print('Episode: ', ep+1, '\t', 'Time: ', time, '\t', 'action_cnt: ',self.action_cnt,'\t','cache_hit: ', self.cache_hit_cnt, '\t', 
            'cache_hit_rate: ', cache_hit_rate, '\t', 'existing_content: ',existing_content, '\t','denominator: ', denominator, '\t','avg_hop: ', avg_hop, '\t',
            'Redundancy: ', redundancy, '\t', 'Reward: ', episode_reward, 'Gamma Reward: ', self.gamma_episode_reward,'\t', 
            'Cache changed cnt: ', self.cache_changed_cnt,'\t','action_3_cnt: ',self.action_3_cnt,'\t', 'qs_action_cnt: ', qs_action_cnt, '\t','random_action_cnt: ', random_action_cnt, '\n')
            
        
            self.write_result_file(ep, time, self.action_cnt, self.cache_hit_cnt, cache_hit_rate, existing_content, denominator, avg_hop, redundancy, 
            episode_reward, self.gamma_episode_reward, self.cache_changed_ratio, self.action_3_cnt, random_action_cnt, qs_action_cnt)
            
            #print(self.roundRequestDictionary)

            self.save_epi_reward.append(episode_reward)
            self.save_epi_cache_hit_rate.append(cache_hit_rate)
            self.save_epi_redundancy.append(redundancy)
            self.save_epi_avg_hop.append(avg_hop)

            ## save weights every episode
            self.dqn.save_weights("./save_weights/cacheSIM_dqn.h5")

        #np.savetxt('./save_weights/cacheSIM_epi_reward.txt', self.save_epi_reward)
        #np.savetxt('./save_weights/cacheSIM_epi_hit.txt', self.save_epi_cache_hit_rate)


    ## save them to file if done
    def plot_result(self):
        self.sf.plot_reward_result(self.save_epi_reward)
        self.sf.plot_cache_hit_result(self.save_epi_cache_hit_rate)
        self.sf.plot_redundancy_result(self.save_epi_redundancy)
        self.sf.plot_existing_content_result(self.save_epi_existing_content)
        self.sf.plot_denominator_result(self.save_epi_denominator)
        self.sf.plot_avg_hop_result(self.save_epi_avg_hop)
        

    def write_result_file(self, ep, time, action_cnt, cache_hit, cache_hit_rate, existing_content, denominator, avg_hop, redundancy, episode_reward, gamma_episode_reward, cache_changed_ratio, action_3_cnt, random_action_cnt, qs_action_cnt):

        if (ep+1)%50 == 0:
            print("===========================50???===========================")
            self.sf.save_result_file()
            #self.sf.write_tmp_reward(self.save_epi_round_reward_d_core, self.save_epi_round_reward_d_cache, self.save_epi_round_reward_c_node, self.save_epi_round_reward_alpha_redundancy, self.save_epi_round_reward_beta_redundancy, self.save_epi_round_reward_vacancy, self.save_epi_round_reward_front, self.save_epi_round_reward_back)

        self.sf.write_result_file(ep, time, action_cnt, cache_hit, cache_hit_rate, existing_content, denominator, avg_hop, redundancy, episode_reward, gamma_episode_reward, cache_changed_ratio, action_3_cnt, random_action_cnt, qs_action_cnt)

    
    def write_actionDictionary_file(self, episode, requestDictionary, actionDictionary):
        self.sf.write_actionDictionary_file(episode, requestDictionary, actionDictionary)
    

    def act(self, path, requested_content, action):

        # self.round_day ???????????? 3??? ????????? requested ??? ????????? ????????? ?????? ??? caching ?????? ?????? ??????
        
        # self.roundRequestDictionary : [???, ???, ???, ???, ???, ???, ???] : ??? ?????? ??? request ?????? ?????????
        # self.roundRequestDictionary = { "title_1" = [0,2,0,14,20,3,4], ... , "title_N" = [30,5,0,0,0,3,8]}
        # round_day ??? 0?????? 0,1,2 ??? ???,???,??? ??? requested ??? ????????? ?????? ????????? ?????????
        date = self.network.days[self.round_nb]
        round_day =  date % 7

        requested_content:ct.Content = requested_content
        requested_content_title = requested_content.get_title()
        tmp_round_request = 0
        roundRequested_cnt_list = []
        path = path

        requested_content_round_cnt = self.roundRequestDictionary[requested_content_title][round_day] + self.roundRequestDictionary[requested_content_title][(round_day+1)%7] + self.roundRequestDictionary[requested_content_title][(round_day+2)%7] + self.roundRequestDictionary[requested_content_title][(round_day+3)%7]
        #print("\n\n")
        #print("requested content : {}".format(requested_content_title))
        #print("round_day : {}".format(round_day))
        #print(requested_content_title + " : " + str(self.roundRequestDictionary[requested_content_title]))
        #print("cnt : {}".format(requested_content_round_cnt))

        # !MicroBS ??? ?????? ---> ??????????????? ?????????(?????? ??????????????? ?????? ?????????) ?????? ??????
        # !????????? ?????? Gain ??? ????????? delete

        # !?????? ??? ?????? ????????? ?????? ????????? ????????? ???????????? ????????? ?????? 
        # !popularity ??????
        if action == 0:
            
            # get_c_node ??? ?????? ??????
            self.stored_type = 0
            self.stored_nodeID = path[1]

            # ????????? ?????? ??????? -> ????????? ????????? ??????? -> 1. ??????. / 2. ?????? ??? ??????.
            if self.network.microBSList[path[1]].storage.isstored(requested_content) != 1:

                if self.network.microBSList[path[1]].storage.abletostore(requested_content):
                    self.network.microBSList[path[1]].storage.addContent(requested_content,date)
                    # cache_changed
                    self.cache_changed = 1
                    self.cache_changed_cnt +=1

                else:
                    for i in range(len(self.network.microBSList[path[1]].storage.storage)):
                        storage_content_title = self.network.microBSList[path[1]].storage.storage[i].get_title()
                        tmp_round_request = self.roundRequestDictionary[storage_content_title][round_day] + self.roundRequestDictionary[storage_content_title][(round_day + 1)%7] + self.roundRequestDictionary[storage_content_title][(round_day + 2)%7] + self.roundRequestDictionary[storage_content_title][(round_day + 3)%7]
                        roundRequested_cnt_list.append(tmp_round_request)
                        #print(str(storage_content_title) + " : " +str(self.roundRequestDictionary[storage_content_title]))
                        #print(tmp_round_request)

                    min_index = roundRequested_cnt_list.index(min(roundRequested_cnt_list))
                    #print("roundRequested_cnt_list : {}".format(roundRequested_cnt_list))
                    #print("min_index : {}".format(min_index))

                    if requested_content_round_cnt > roundRequested_cnt_list[min_index]:
                        
                        del_content = self.network.microBSList[path[1]].storage.storage[min_index]


                        #print("del_content : {}".format(del_content.__dict__))
                        #print("????????? ???")
                        #for j in range(len(self.network.microBSList[path[1]].storage.storage)):
                        #    print(self.network.microBSList[path[1]].storage.storage[j].__dict__)

                        self.network.microBSList[path[1]].storage.delContent(del_content)
                        self.network.microBSList[path[1]].storage.addContent(requested_content,date)


                        #print("?????? ???")
                        #for j in range(len(self.network.microBSList[path[1]].storage.storage)):
                        #    print(self.network.microBSList[path[1]].storage.storage[j].__dict__)

                        self.cache_changed = 1
                        self.cache_changed_cnt +=1


        # BS ??? ?????? ---> ??????????????? ????????? ?????? ??????
        elif action == 1:
            
            # get_c_node ??? ?????? ??????
            self.stored_type = 1
            self.stored_nodeID = path[2]

            if self.network.BSList[path[2]].storage.isstored(requested_content) != 1:
                if self.network.BSList[path[2]].storage.abletostore(requested_content):
                    self.network.BSList[path[2]].storage.addContent(requested_content,date)
                    # cache_changed
                    self.cache_changed = 1
                    self.cache_changed_cnt +=1

                else:
                    for i in range(len(self.network.BSList[path[2]].storage.storage)):
                        storage_content_title = self.network.BSList[path[2]].storage.storage[i].get_title()
                        tmp_round_request = self.roundRequestDictionary[storage_content_title][round_day] + self.roundRequestDictionary[storage_content_title][(round_day + 1)%7] + self.roundRequestDictionary[storage_content_title][(round_day + 2)%7] + self.roundRequestDictionary[storage_content_title][(round_day + 3)%7]
                        roundRequested_cnt_list.append(tmp_round_request)
                        #print(str(storage_content_title) + " : " +str(self.roundRequestDictionary[storage_content_title]))
                        #print(tmp_round_request)

                    min_index = roundRequested_cnt_list.index(min(roundRequested_cnt_list))
                    #print("roundRequested_cnt_list : {}".format(roundRequested_cnt_list))
                    #print("min_index : {}".format(min_index))

                    if requested_content_round_cnt > roundRequested_cnt_list[min_index]:
                        
                        del_content = self.network.BSList[path[2]].storage.storage[min_index]
                        
                        #print("del_content : {}".format(del_content.__dict__))
                        #print("????????? ???")
                        #for j in range(len(self.network.BSList[path[2]].storage.storage)):
                        #    print(self.network.BSList[path[2]].storage.storage[j].__dict__)

                        self.network.BSList[path[2]].storage.delContent(del_content)
                        self.network.BSList[path[2]].storage.addContent(requested_content,date)

                        #print("?????? ???")
                        #for j in range(len(self.network.BSList[path[2]].storage.storage)):
                        #    print(self.network.BSList[path[2]].storage.storage[j].__dict__)

                        self.cache_changed = 1
                        self.cache_changed_cnt +=1



        # DataCenter ??? ?????? ---> ??????????????? ????????? ?????? ??????
        elif action == 2:

            # get_c_node ??? ?????? ??????
            self.stored_type = 2
            self.stored_nodeID = path[3]

            if self.network.dataCenter.storage.isstored(requested_content) != 1:
                if self.network.dataCenter.storage.abletostore(requested_content):
                    self.network.dataCenter.storage.addContent(requested_content,date)
                    # cache_changed
                    self.cache_changed = 1
                    self.cache_changed_cnt +=1
                else:
                    for i in range(len(self.network.dataCenter.storage.storage)):
                        storage_content_title = self.network.dataCenter.storage.storage[i].get_title()
                        tmp_round_request = self.roundRequestDictionary[storage_content_title][round_day] + self.roundRequestDictionary[storage_content_title][(round_day + 1)%7] + self.roundRequestDictionary[storage_content_title][(round_day + 2)%7] + self.roundRequestDictionary[storage_content_title][(round_day + 3)%7]
                        roundRequested_cnt_list.append(tmp_round_request)
                        #print(str(storage_content_title) + " : " +str(self.roundRequestDictionary[storage_content_title]))
                        #print(tmp_round_request)

                    min_index = roundRequested_cnt_list.index(min(roundRequested_cnt_list))
                    #print("roundRequested_cnt_list : {}".format(roundRequested_cnt_list))
                    #print("min_index : {}".format(min_index))

                    if requested_content_round_cnt > roundRequested_cnt_list[min_index]:
                        
                        del_content = self.network.dataCenter.storage.storage[min_index]

                        #print("del_content : {}".format(del_content.__dict__))
                        #print("????????? ???")
                        #for j in range(len(self.network.dataCenter.storage.storage)):
                        #    print(self.network.dataCenter.storage.storage[j].__dict__)

                        self.network.dataCenter.storage.delContent(del_content)
                        self.network.dataCenter.storage.addContent(requested_content,date)

                        #print("?????? ???")
                        #for j in range(len(self.network.dataCenter.storage.storage)):
                        #    print(self.network.dataCenter.storage.storage[j].__dict__)

                        self.cache_changed = 1
                        self.cache_changed_cnt +=1
        
        elif action == 3:
            self.action_3_cnt += 1
     
        

    def get_reward(self, action, path, requested_content):
        """
        Return the reward.
        The reward is:
        
            Reward = a*(d_core - d_cache) - b*(#ofnode - coverage_node)

            a,b = ????????? ???????????? ??????????????? ?????????
            d_core  : ???????????? ???????????? ?????? ???????????? ?????? ?????? ????????? ???????????? ?????? ??????.
            d_cache : ?????? ????????? ????????? ?????? ???????????? ?????? ???????????? ????????? ??? ????????? ?????? ?????? ??????
            cf.NB_NODES : ????????? ??????
            c_node : agent ????????? ??? contents??? ?????? station??? ???????????? device??? ??????
        """
        
        #reward = 0
        self.set_reward_parameter(path, requested_content=requested_content)

        if self.cache_changed == 0:

            if action == 3:
                reward = -1*self.e*self.vacancy

            # action = 0,1,2 cache_changed == 0
            else:
                reward = -1*(self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node) - self.c*self.alpha_redundancy - self.d*self.beta_redundancy) - self.e*self.vacancy
     
        # action 0,1,2 cache_changed == 1
        else:
            reward = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node) - self.c*self.alpha_redundancy - self.d*self.beta_redundancy - self.e*self.vacancy
        #          1         20                              0.5               300         0.5        300                   0.1       0                   10    
               
            #front = self.a*(self.d_core - self.d_cache) + self.b*math.log2(self.c_node) - self.c*self.alpha_redundancy - self.d*self.beta_redundancy
            #back = self.e*self.vacancy
            
            #self.save_epi_round_reward_d_core.append(self.d_core)
            #self.save_epi_round_reward_d_cache.append(self.d_cache)
            #self.save_epi_round_reward_c_node.append(self.c_node)
            #self.save_epi_round_reward_alpha_redundancy.append(self.alpha_redundancy)
            #self.save_epi_round_reward_beta_redundancy.append(self.beta_redundancy)
            #self.save_epi_round_reward_vacancy.append(self.vacancy)
            #self.save_epi_round_reward_front.append(front)
            #self.save_epi_round_reward_back.append(back)

       

        """
        print("self.d_core : {}".format(self.d_core))
        print("self.d_cache : {}".format(self.d_cache))
        print("self.c_node : {}".format(self.c_node))
        print("self.alpha : {}".format(self.alpha_redundancy))
        print("self.beta : {}".format(self.beta_redundancy))
        """
        # cache_changed ?????????
        self.cache_changed = 0
        reward = float(reward)
        #print(reward)
        return reward

    def set_reward_parameter(self, path, requested_content):

        # d_core  : ???????????? ???????????? ?????? ???????????? ?????? ?????? ????????? ???????????? ?????? ??????.
        #          

        # d_cache : ?????? ????????? ????????? ?????? ???????????? ?????? ???????????? ????????? ??? ????????? ?????? ?????? ??????
                   
        # R_cache : ??????????????? ???????????? ????????? ????????? ???
        #           for ????????? ???????????????

        # c_node   : ????????? ????????? ???????????? ????????? ??? (coverage node)
        #           
        nodeID = path[0]
        self.d_core = self.get_d_core(nodeID, requested_content)
        self.d_cache = self.get_d_cache(nodeID, requested_content)
        self.c_node = self.get_c_node()
        self.alpha_redundancy, self.beta_redundancy = self.set_content_redundancy(requested_content)
        self.vacancy = self.cal_vacancy()

        #self.run_TwinNetwork()


    def get_d_core(self,nodeID, requested_content):
        # ?????? ??????????????? ?????? ???????????? ????????? ?????????
        # path ?????? ???????????? ?????????
        path = []
        path = self.network.requested_content_and_get_path(nodeID, requested_content)

        # [4,68] ??? ?????? ---> [4,68, search_next_path(microBS.x, microBS.y):BS, search_next_path(BS.x, BS.y):Datacenter, search_next_path(Datacenter.x, Datacenter.y):Core Internet]
        # path ??? ????????? ????????? ?????????
        while len(path) != 5:

            # Micro??? ???????????? ?????? ??????, BS ??????
            if len(path) == 2:
                id = path[-1]
                closestID = self.network.search_next_path(self.network.microBSList[id].pos_x,self.network.microBSList[id].pos_y,1)
                path.append(closestID)

            # BS??? ?????? ?????? ?????? ??????, Data Center ??????
            elif len(path) ==  3:
                path.append(0)

            # ????????? ????????? ????????? ?????? ?????? ??????, Core Internet ??????
            elif len(path) == 4:
                path.append(0)
        #print(self.network.uplink_latency(path).shape)
        d_core = (self.network.uplink_latency(path) + self.network.downlink_latency(path)) * 1000
        
        return d_core

    def get_d_cache(self, nodeID, requested_content):
        # TODO : ?????? ????????? ????????? ?????? ???????????? ?????? ???????????? ????????? ??? ????????? ?????? ?????? ??????
        path = []
        path = self.network.requested_content_and_get_path(nodeID, requested_content)

        d_cache = (self.network.uplink_latency(path) + self.network.downlink_latency(path)) * 1000

        return d_cache

    def get_c_node(self):
        # TODO : agent ????????? ??? contents??? ?????? station??? ???????????? device??? ??????
        c_node = 0
        tmpcnt = 0

        # MicroBS
        if self.stored_type == 0:
            c_node = len(self.network.MicroBSNodeList[self.stored_nodeID])
        
        # BS
        elif self.stored_type == 1:
            for i in self.network.BSNodeList[self.stored_nodeID]:
                tmpcnt += len(self.network.MicroBSNodeList[i])
            c_node = tmpcnt

        # DataCenter
        elif self.stored_type == 2:
            c_node = cf.NB_NODES

        return c_node

    def set_content_redundancy(self, content):
        
        #print("requested_content : {}".format(content.__dict__))
        full_redundancy = self.cal_full_redundancy(content)
        alpha_redundancy = self.cal_alpha_redundancy(content)
        beta_redundancy = full_redundancy - alpha_redundancy

        #print("full_redundancy : {}".format(full_redundancy))
        #print("alpha_redundancy : {}".format(alpha_redundancy))
        #print("full_redundancy - alpha_redundancy : {}".format(full_redundancy - alpha_redundancy))
        #print("beta_redundancy : {}".format(beta_redundancy))

        return alpha_redundancy, beta_redundancy

    # cal_alpha_redundacy
    # agent??? ????????? ?????? ?????? ????????? content redundancy ??? ??????
    def cal_alpha_redundancy(self, content):
        # ?????? ?????? ??????
        content_redundancy = 1

        """
        # @ agent??? ????????? ?????? Micro BS ?????? ?????? ????????? ?????? ????????? ?????? X
        if self.stored_type == 0:
            content_redundancy = content_redundancy + 1
        """    

        # @ agent??? ????????? ?????? Base Station ?????? ?????? Micro Node??? content redundancy ??????
        if self.stored_type == 1:
            leaf_node_list = self.network.BSNodeList[self.stored_nodeID]
            #print('microBS')
            #print(leaf_node_list)
            for i in leaf_node_list:
                if content in self.network.microBSList[i].storage.storage:
                    content_redundancy = content_redundancy + 1

        # @ agent??? ????????? ?????? Data Center?????? ?????? MicroBS, BS???  content redundancy ????????????
        elif self.stored_type == 2:
            # Micro BS
            for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
                if content in self.network.microBSList[i].storage.storage:
                    content_redundancy = content_redundancy + 1
            # BS
            for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
                if content in self.network.BSList[i].storage.storage:
                    content_redundancy = content_redundancy + 1

        return content_redundancy

    def cal_full_redundancy(self, content):
        full_redundancy = 0
        # Micro BS
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):

            if content in self.network.microBSList[i].storage.storage:
                full_redundancy = full_redundancy + 1

            
            #print("{}?????? MicroBase Station Storage".format(i))
            #for j in range(len(self.network.microBSList[i].storage.storage)):
            #    print(self.network.microBSList[i].storage.storage[j].__dict__)
            
            
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):

            if content in self.network.BSList[i].storage.storage:
                full_redundancy = full_redundancy + 1

            
            #print("{}?????? Base Station Storage".format(i)) 
            #for j in range(len(self.network.BSList[i].storage.storage)):
            #    print(self.network.BSList[i].storage.storage[j].__dict__)
            

        # Datacenter
        if content in self.network.dataCenter.storage.storage:
            full_redundancy = full_redundancy + 1

        #print("Datacenter Storage")        
        #for j in range(len(self.network.dataCenter.storage.storage)):
        #    print(self.network.dataCenter.storage.storage[j].__dict__)
    
        return full_redundancy
 
#@  ?????????( ?????? ????????? ?????? -1 ) / ?????? ???????????? ????????? ??????
#@  ????????? ?????? ???????????? ????????????
#@  ????????? network.py ?????? ?????????????????? 
#@   reward ????????? ?????? ????????? ?????? ?????? ??? ??????
#@   list??? ?????? ????????? ??? ????????????
    def function1(self):

        title_list = sc.emBBScenario.titleList
        #print(title_list)
        contentdict = {}

        for i in range(len(title_list)):
            contentdict[title_list[i]] = 0

        for title in title_list:
            #print("???????????? title : {}".format(title))
            # Micro BS
            for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
                for content in self.network.microBSList[i].storage.storage:
                    if title == content.get_title():
                        #print("content.get_title() : {}".format(content.get_title()))
                        contentdict[title] = contentdict[title] + 1

            # BS
            for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
                for content in self.network.BSList[i].storage.storage:
                    if title == content.get_title():
                        #print("content.get_title() : {}".format(content.get_title()))
                        contentdict[title] = contentdict[title] + 1

            # DataCenter
            for content in self.network.dataCenter.storage.storage:
                if title == content.get_title():
                    #print("content.get_title() : {}".format(content.get_title()))
                    contentdict[title] = contentdict[title] + 1

        #print(contentdict)
        
        denominator = 0
        existing_content = 0
        result = 0

        for title in title_list:
            if contentdict[title] != 0:
                existing_content += 1
                denominator += contentdict[title]

        print("denominator - existing_content : {} ".format(denominator - existing_content))
        print("existing_content : {} ".format(existing_content))
        print("denominator : {} ".format(denominator))

        self.save_epi_denominator.append(denominator)
        self.save_epi_existing_content.append(existing_content)

        result = (denominator - existing_content) / denominator

        print("?????????( ?????? ????????? ?????? - 1 ) / ?????? ???????????? ????????? ?????? : {}".format(result))

        return result, existing_content, denominator

    def set_roundRequestDictionary(self):
        title_list = sc.emBBScenario.titleList
        roundRequestdict = {}

        for i in range(len(title_list)):
            # [??? ??? ??? ??? ??? ??? ???]
            # request ??? ?????? plot ??? ??????
            roundRequestdict[title_list[i]] = [0,0,0,0,0,0,0]
        return roundRequestdict

    def updateRoundRequestDictionary(self, title, round_day):
        #print(title + ' , ' + str(round_day))
        self.roundRequestDictionary[title][round_day] += 1
        #print(self.roundRequestDictionary)

    def set_contentLabelDictionary(self):
        title_list = sc.emBBScenario.titleList
        dict = {}

        for i in range(len(title_list)):
            dict[title_list[i]] = i+1

        #print(dict)
        return dict

    def set_requestDictionary(self):
        title_list = sc.emBBScenario.titleList
        #print(title_list)
        contentdict = {}

        for i in range(len(title_list)):
            contentdict[title_list[i]] = 0

        return contentdict

    def update_requestDictionary(self, title):
        self.requestDictionary[title] += 1

    def set_actionDictionary(self):
        title_list = sc.emBBScenario.titleList
        #print(title_list)
        actiondict = {}

        for i in range(len(title_list)):
            actiondict[title_list[i]] = []

        return actiondict

    def append_actionDictionary(self, title, action):
        self.actionDictionary[title].append(action)

    def cal_vacancy(self):

        vacancy = 0 
        #print("=========================MicroBS=========================")
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            #print(self.network.microBSList[i].storage.__dict__)
            #print("vacancy : {}".format(self.network.microBSList[i].storage.capacity - self.network.microBSList[i].storage.stored))
            vacancy += self.network.microBSList[i].storage.capacity - self.network.microBSList[i].storage.stored

        #print("=========================BS=========================")
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            #print(self.network.BSList[i].storage.__dict__)
            #print("vacancy : {}".format(self.network.BSList[i].storage.capacity - self.network.BSList[i].storage.stored))
            vacancy += self.network.BSList[i].storage.capacity - self.network.BSList[i].storage.stored

        #print("=========================Datacenter=========================")
        #print(self.network.dataCenter.storage.__dict__)
        #print("vacancy : {}".format(self.network.dataCenter.storage.capacity - self.network.dataCenter.storage.stored))
        vacancy += self.network.dataCenter.storage.capacity - self.network.dataCenter.storage.stored

        return vacancy
        
         
    def showAllStorage(self, network):
        print("===============SHOW ALL STORAGE===============")
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            print("{}?????? Micro BS Storage".format(i))

            for j in range(len(network.microBSList[i].storage.storage)):
                content = network.microBSList[i].storage.storage[j]
                print(str(content.__dict__), " ==> lastdate : ", str(network.microBSList[i].storage.lastdatelist[j]))
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            print("{}?????? BS Storage".format(i))
            for j in range(len(network.BSList[i].storage.storage)):
                content = network.BSList[i].storage.storage[j]
                print(str(content.__dict__), " ==> lastdate : ", str(network.BSList[i].storage.lastdatelist[j]))
        # DataCenter
        print("DataCenter Storage")
        for j in range(len(network.dataCenter.storage.storage)):
            content = network.dataCenter.storage.storage[j]
            print(str(content.__dict__), " ==> lastdate : ", str(network.dataCenter.storage.lastdatelist[j]))

        print("===============FINISH SHOW ALL STORAGE===============")

                        
    def deleteOutofdateContents(self):
        
        limitdate = self.date - cf.DELETE_PERIOD
        delcontentlist = []
        #MicroBS
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            
            for j in range(len(self.network.microBSList[i].storage.storage)):
                content = self.network.microBSList[i].storage.storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.network.microBSList[i].storage.lastdatelist[j]))
                contentlastdate = self.network.microBSList[i].storage.lastdatelist[j]
                if contentlastdate < limitdate:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.network.microBSList[i].storage.delContent(delcontent)

            delcontentlist = []
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            
            for j in range(len(self.network.BSList[i].storage.storage)):
                content = self.network.BSList[i].storage.storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.network.BSList[i].storage.lastdatelist[j]))
                contentlastdate = self.network.BSList[i].storage.lastdatelist[j]
                
                if contentlastdate < limitdate:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.network.BSList[i].storage.delContent(delcontent)
            delcontentlist = []

        # DataCenter
        for j in range(len(self.network.dataCenter.storage.storage)):
            content = self.network.dataCenter.storage.storage[j]
            #print(str(content.__dict__), " ==> lastdate : ", str(self.network.dataCenter.storage.lastdatelist[j]))
            contentlastdate = self.network.dataCenter.storage.lastdatelist[j]
            if contentlastdate < limitdate:
                delcontentlist.append(content)
                
        for delcontent in delcontentlist:
            self.network.dataCenter.storage.delContent(delcontent)
        delcontentlist = []

    def set_qs_action_count_list(self):
        self.qs_action_count_list = [0,0,0,0]

    def set_random_action_count_list(self):
        self.random_action_count_list = [0,0,0,0]


    def run_TwinNetwork(self):
        
        # cf.TWIN ROUND ?????? ?????????
        # ?????? original network??? ?????????
        # ????????? ??? round_day + 1 ?????????
        # return total latency, cache hit rate

        # @ ?????? ?????? 

        self.twin_network = copy.deepcopy(self.network)        
        twin_round_nb = self.round_nb
        total_latency = 0
        cache_hit_cnt = 0
        cache_hit_rate = 0 
        lastdate = self.twin_network.days[twin_round_nb - 1]

        for round in range(cf.TWIN_ROUND):

            #print("Twin round : {}".format(round))

            date = self.twin_network.days[twin_round_nb]
            round_day = date%7
            requested_content, path = self.twin_network.request_and_get_path(round_day)
            #print(requested_content.__dict__)
            #print(path)
            total_latency += (self.twin_network.uplink_latency(path) + self.twin_network.downlink_latency(path)) * 1000

            if date != lastdate:
                print("======================================================DATE CHANGED======================================================")
                self.deleteOutofdateContentsTwinNetwork()
        
            # Cache hit : Datacenter || CoreNetwork
            if len(path) >= 4:
                # DataCenter cache hit                
                if len(path) == 4:
                    cache_hit_cnt += 1

            # Cache hit : MicroBS, BS
            else:
                cache_hit_cnt += 1
            
            lastdate = date
            twin_round_nb = twin_round_nb + 1
            
        cache_hit_rate = cache_hit_cnt/cf.TWIN_ROUND

        print(total_latency)
        print(cache_hit_rate)
        #print("==================================self.network storage==================================")
        #self.showAllStorage(self.network)
        #print("==================================self.twin_network storage==================================")
        #self.showAllStorage(self.twin_network)

        return total_latency, cache_hit_rate

    def deleteOutofdateContentsTwinNetwork(self):
        
        limitdate = self.date - 1
        delcontentlist = []
        #MicroBS
        for i in range(cf.NUM_microBS[0]*cf.NUM_microBS[1]):
            
            for j in range(len(self.twin_network.microBSList[i].storage.storage)):
                content = self.twin_network.microBSList[i].storage.storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.twin_network.microBSList[i].storage.lastdatelist[j]))
                contentlastdate = self.twin_network.microBSList[i].storage.lastdatelist[j]
                if contentlastdate < limitdate:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.twin_network.microBSList[i].storage.delContent(delcontent)

            delcontentlist = []
        # BS
        for i in range(cf.NUM_BS[0]*cf.NUM_BS[1]):
            
            for j in range(len(self.twin_network.BSList[i].storage.storage)):
                content = self.twin_network.BSList[i].storage.storage[j]
                #print(str(content.__dict__), " ==> lastdate : ", str(self.twin_network.BSList[i].storage.lastdatelist[j]))
                contentlastdate = self.twin_network.BSList[i].storage.lastdatelist[j]
                
                if contentlastdate < limitdate:
                    delcontentlist.append(content)
                
            for delcontent in delcontentlist:
                self.twin_network.BSList[i].storage.delContent(delcontent)
            delcontentlist = []

        # DataCenter
        for j in range(len(self.twin_network.dataCenter.storage.storage)):
            content = self.twin_network.dataCenter.storage.storage[j]
            #print(str(content.__dict__), " ==> lastdate : ", str(self.twin_network.dataCenter.storage.lastdatelist[j]))
            contentlastdate = self.twin_network.dataCenter.storage.lastdatelist[j]
            if contentlastdate < limitdate:
                delcontentlist.append(content)
                
        for delcontent in delcontentlist:
            self.twin_network.dataCenter.storage.delContent(delcontent)
        delcontentlist = []
                        



    """
    #! H_arg ??? ?????? ?????? ????????? ?????? ????????? ?????????
    #! get_R_cache ??? get_H_arg ??? ???????????? ????????? ????????????. (2022/05/11)


    def set_reward(self):
    
    # Return the reward.
    # The reward is:
    
    #    Reward = a*(d_core - d_cache) - b*(R_cache/H_arg)

    #    a,b = ????????? ???????????? ??????????????? ?????????

    #    d_core  : ???????????? ???????????? ?????? ???????????? ?????? ?????? ????????? ???????????? ?????? ??????.
    #    d_cache : ?????? ????????? ????????? ?????? ???????????? ?????? ???????????? ????????? ??? ????????? ?????? ?????? ??????

    #    ! ?????????  
    #    R_cache : ??????????????? ???????????? ????????? ????????? ???
    #    H_arg   : ????????? ?????? ????????? ?????? ??? ??? (????????? ???????????? ?????????)
    
    self.reward = 0


    self.reward = self.a*(self.d_core - self.d_cache) - self.b*(self.R_cache/self.H_arg)
    return self.reward

    def get_reward_parameter(self):

        # d_core  : ???????????? ???????????? ?????? ???????????? ?????? ?????? ????????? ???????????? ?????? ??????.
        #           ?????? ??????

        # d_cache : ?????? ????????? ????????? ?????? ???????????? ?????? ???????????? ????????? ??? ????????? ?????? ?????? ??????
                   
        # R_cache : ??????????????? ???????????? ????????? ????????? ???
        #           for ????????? ???????????????

        # H_arg   : ????????? ?????? ????????? ?????? ??? ??? (????????? ???????????? ?????????)
        #           ?????????

        return d_core, d_cache, R_cache, H_arg

    # R_cache ?????????
    def get_R_cache(self, content:ct.Content):

        # ????????? ????????? ???????????? ????????? R_cache ?????????
        self.R_cache = 0

        # INFO : network.storage : [capacity , stored_size, [contents:contentStorage]]
        #        network.storage.storage : [{'title' : '??????', 'size' : 123}, ... ,{'title' : '9?????????', 'size' : 123}]

        # TODO : 1. Data Center ??? ?????? ??? ??????
        
        if content in self.network.dataCenter.storage.storage:

            self.R_cache = self.R_cache + 1

        # TODO : 2. Base Station ??? ?????? ??? ??????
        # 2.1 Base Station List ?????????
        for i in self.network.BSList:

            # 2.2 ?????? Base Station??? stored ??? contents List??? ?????????
            storedContentsList:List = i.storage.storage

            if content in storedContentsList:
                self.R_cache = self.R_cache + 1


        # TODO : 3. Micro Base Station ??? ?????? ??? ??????
        for i in self.network.microBSList:

            # 2.2 ?????? Base Station??? stored ??? contents List??? ?????????
            storedContentsList:List = i.storage.storage

            if content in storedContentsList:
                self.R_cache = self.R_cache + 1

        return self.R_cache

    def get_H_arg():


    """
