import random
import numpy as np
import math
import Complex_Parameter_Set as PS
from matplotlib import pyplot as plt
import Complex_support as support
import Complex_deep_support as dp
import tensorflow as tf
import time

import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(PS.length)
#
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0],enable=True)



'''
 #测试代码
from support import Satellite
from support import User
satellite = Satellite()
user_1 = User()
#user_2 = User()
#设定仰角为40度
user_1.position = [0.0,0.0]
#user_2.position = [0.268,0.0]
satellite.position = [0.0,0.0]
SINR = dp.gen_SNR(satellite,user_1)
print(SINR)
print(dp.rate(SINR))
'''
'''
SNR = []
for x in range(300):
    SNR.append(dp.gen_SNR(satellite, user))
'''
# print(dp.cal_off_axis(satellite,user_1,user_2))
# print(dp.cal_elevation(satellite, user_1))
# print(dp.cal_elevation(satellite, user_2))

#print(np.mean(SNR))
#plt.plot(SNR)
#plt.show()

# mean_new_call = []
# mean_block = []
# mean_reward = []
# mean_packets = []
# mean_cost = []
# mean_handovertimes = []
# mean_non_handovertimes = []

# network = dp.network()
network_set = [dp.network() for i in range(PS.user_num)]        
count = support.Count()     

# network = dp.network()
# model = tf.saved_model.load("saved/c_max_50")
# assert PS.C_max == 50,'Wrong!'
# network.evaluate_model = model

pro = 1 - math.exp(-PS.lamda * PS.slot)         

#
# X = []
# Y = []
# for user in user_set:
#     X.append(user.position[0])
#     Y.append(user.position[1])
# plt.plot(X,Y,'r*')
# plt.show()


for episode_id in range(PS.num_episode):
    print(episode_id)
    print(time.asctime(time.localtime(time.time())))
#    count.episode_num = episode_id     

    user_set = support.gen_User_Set()
    support.gen_position(user_set)
    support.gen_neighborhood(user_set)

    # X = []
    # Y = []
    # for user in user_set:
    #     X.append(user.position[0])
    #     Y.append(user.position[1])
    # plt.plot(X,Y,'r*')
    # plt.show()

    for index,user in enumerate(user_set):
        user.network = network_set[index]            
    constellation = support.gen_Constellation()
    epsilon = max(  
        PS.initial_epsilon * (PS.num_exploration_episodes - episode_id) / PS.num_exploration_episodes,
        PS.final_epsilon)
    beta = max(  
        PS.initial_beta * (PS.num_exploration_episodes - episode_id) / PS.num_exploration_episodes,
        PS.final_beta)
    N_slot = 0
    t = N_slot * PS.slot

    for user in user_set:
        user.candidate_update(constellation,N_slot)

    while t < PS.T:
        N_slot += 1
        t = N_slot * PS.slot
        support.constellation_move(constellation)
        for user in user_set:       
            user.epsilon = epsilon
            user.beta = beta
            user.position_update()

            user.old_traffic = user.traffic
            if user.traffic == 'OFF':
                if random.random() < pro:      
                    user.traffic = 'ON'
                    user.ON_time = support.gen_ON_time()
                    user.new_access(constellation,count)     
            else:
                count.sum_ON_time += 1     
                user.ON_time -= 1
                if user.ON_time == 0:
                    user.traffic = 'OFF'
                    user.ON_time = None
                    user.shut_down(constellation)      

        if (N_slot % PS.T_H == 0):
            print(N_slot)
            print(time.asctime(time.localtime(time.time())))
            for user in user_set:
                user.candidate_update(constellation,N_slot)
                
        if N_slot % PS.T_H == PS.T_D:       
            support.system_handover(user_set, constellation, count)  

        support.system_normal(user_set,constellation,count)
    count.reset()
    print('ON_time = ', count.ON_time_per_episode)
    print('reward = ', count.reward_per_episode)
    print('cost = ', count.cost_per_episode)
    print('throughput = ', count.throughput_per_episode)
    print('rate = ', count.average_rate_per_episode)
    print('handover_times = ', count.handovertimes_per_episode)
    print('fail_times = ', count.failtimes_per_episode)
    print('success_times = ', count.successtimes_per_episode)
    print('waiting_times = ', count.waiting_times_per_episode)
    print('sum_waiting_user =', count.waiting_user_per_episode)
    print('sum_active_user =', count.active_user_per_episode)
    print('reward_2 = ', count.reward_2_per_episode)

C_max = PS.C_max
C_max = str(C_max)

savepath = 'saved/' + C_max + '/'
for index_i,network in enumerate(network_set):
    savename = savepath + str(index_i)
    tf.keras.models.save_model(network.evaluate_model,savename)