import numpy as np
import Complex_Parameter_Set as PS
import math
import random
import Complex_deep_support as dp
import copy
import Position


def gen_ON_time():
    t = -PS.T_m * math.log(1 - random.random())
    N_slot = round(t/PS.slot)
    return N_slot

class Count(object):
    def __init__(self):
#        self.episode_num = None
        self.sum_ON_time = 0  
        self.sum_active_user = 0 
        self.sum_waiting_user = 0  
        self.sum_reward = 0.0  
        self.sum_reward_2 = 0.0
        self.sum_cost = 0.0  
        self.sum_throughput = 0.0  
        self.sum_handovertimes = 0  
        self.sum_failtimes = 0  
        self.sum_successtimes = 0 
        self.sum_waiting_times = 0

        self.ON_time_per_episode = []
        self.active_user_per_episode = []
        self.waiting_user_per_episode = []
        self.reward_per_episode = []
        self.reward_2_per_episode = []
        self.cost_per_episode = []
        self.throughput_per_episode = []
        self.average_rate_per_episode = []
        self.handovertimes_per_episode = []
        self.failtimes_per_episode = []
        self.successtimes_per_episode = []
        self.waiting_times_per_episode = []

    def reset(self):
        average_reward = self.sum_reward / int(PS.T / PS.slot)  
        average_reward_2 = self.sum_reward_2 / int(PS.T / PS.slot)
        average_throughput = self.sum_throughput * PS.packet_size * 1e-6 / PS.T  
        average_rate = self.sum_throughput * PS.packet_size * 1e-6 / (self.sum_active_user * PS.slot)  
        average_cost = self.sum_cost / self.sum_active_user 
        average_active_user = self.sum_active_user / int(PS.T / PS.slot)
        average_waiting_user = self.sum_waiting_user / int(PS.T / PS.slot)

        self.ON_time_per_episode.append(self.sum_ON_time)  
        self.reward_per_episode.append(average_reward)  
        self.reward_2_per_episode.append(average_reward_2)
        self.cost_per_episode.append(average_cost)  
        self.throughput_per_episode.append(average_throughput)  
        self.average_rate_per_episode.append(average_rate)  
        self.handovertimes_per_episode.append(self.sum_handovertimes) 
        self.failtimes_per_episode.append(self.sum_failtimes)  
        self.successtimes_per_episode.append(self.sum_successtimes)  
        self.waiting_times_per_episode.append(self.sum_waiting_times)
        self.active_user_per_episode.append(average_active_user)
        self.waiting_user_per_episode.append(average_waiting_user)


        self.sum_ON_time = 0  
        self.sum_reward = 0.0  
        self.sum_reward_2 = 0.0
        self.sum_cost = 0.0  
        self.sum_throughput = 0.0  
        self.sum_handovertimes = 0  
        self.sum_failtimes = 0  
        self.sum_successtimes = 0  
        self.sum_waiting_times = 0
        self.sum_active_user = 0
        self.sum_waiting_user = 0

class User(object):
    def __init__(self):
        self.position = None      
        self.neighborhood = []
        self.distance = []
        self.traffic = 'OFF'
        self.old_traffic = 'OFF'
        self.ON_time = None
        self.waiting_satellite = None   
        self.access = None
        self.action = None
        self.candidate = [-1 for x in range(PS.candidate_num)]
        self.elevation = [(0.0,0.0,0.0) for x in range(PS.candidate_num)]
        self.condition = [-1 for x in range(PS.candidate_num)]      
#        self.timer = None      
        self.old_state = None
        self.access_flag = None
        self.access_timer = None
        self.cost = 0.0
        self.reward = []
        self.packets_now = None     
        self.epsilon = None
        self.beta = None
        self.network = dp.network()
#        self.new_call = 0       
#        self.block = 0          
#        self.docu_reward = []
#        self.docu_packets = []
#        self.docu_cost = []
#        self.docu_channel = []
#        self.docu_handover = 0
#        self.docu_non_handover = 0

    def position_update(self):
        self.position[0] += (PS.we * PS.slot)     

    # def traffic_updata(self,t):         
    #     slot = round(t/PS.slot)
    #     for (x,y,z) in self.schedule:
    #         if y == slot:
    #             self.traffic = z

    def candidate_update(self,constellation,N_slot):       
        old_set = set(self.candidate)  
        new_set = set()
        for index, x in enumerate(constellation):
            assert index == x.index, '编号出现错误'
            if (dp.cal_elevation(x, self) > PS.theta_min):
                new_set.add(index)
        delta_set = new_set.difference(old_set)  
        delta_set = list(delta_set)  
        L = []
        for index, a in enumerate(self.candidate):
            if a == -1:
                L.append(index)  
        assert len(L) > 0,print('L = ',L)
        L = random.sample(L, len(delta_set)) 
        assert len(L) == len(delta_set), 'len(L) != len(delta_set)'
        for n in range(len(delta_set)):
            self.candidate[L[n]] = delta_set[n]
        for index, x in enumerate(self.candidate):
            if not (dp.cal_elevation(constellation[x], self) > PS.theta_min) :
                self.candidate[index] = -1
        new_candidate = copy.deepcopy(self.candidate)
        self.elevation_update(new_candidate,constellation,N_slot)       


    def elevation_update(self,new_candidate,constellation,N_slot):
        old_elevation = copy.deepcopy(self.elevation)
        new_elevation = []
        for index,x in enumerate(new_candidate):
            if x == -1:
                new_elevation.append((0.0,0.0,0.0))
            else:
                theta = dp.cal_elevation(constellation[x],self)
                theta_t_1 = old_elevation[index][0]
                theta_t_2 = old_elevation[index][1]
                new_elevation.append((theta,theta_t_1,theta_t_2))
        self.elevation = new_elevation
        self.condition_update(new_candidate,N_slot)

    def condition_update(self,new_candidate,N_slot):
        old_condition = copy.deepcopy(self.condition)
        new_condition = []
        for index,x in enumerate(new_candidate):
            if x == -1:
                new_condition.append(-1)
            else:
                new_condition.append(dp.condition_trans(old_condition[index],self.elevation[index][0],N_slot))
        self.condition = new_condition


    def new_access(self,constellation,count):    
        #此步需要更新：candidate,action,old_state,timer,access_flag,access_timer,reward,SINR
        assert self.access == None ,'Access is not None!'
        assert self.action == None, 'The user has chosen a satellite!'
        #self.candidate_update(constellation)       

        candidate_set = set(self.candidate)
        if len(candidate_set) == 1:            
            assert list(candidate_set)[0] == -1, '候选集出错'
            print('真的没有可接入的卫星！！！')
            #self.old_state = None
            self.action = None
            #satellite_num = -1
        else:
            self.old_state = dp.state_output(self,constellation)       
            self.action = dp.predict(self)      
            self.old_state = None              
            #satellite_num = self.candidate[self.action]

        satellite_num = self.candidate[self.action]
        if constellation[satellite_num].channel_overhead < PS.C_max:
            count.sum_handovertimes += 1   
            constellation[satellite_num].channel_overhead += 1
            self.cost = PS.Bc * dp.U_C(constellation[satellite_num].channel_overhead)
            self.access_flag = True
            self.access = None
            self.access_timer = PS.T_h
        else:
            constellation[satellite_num].queue.append(self) 
            self.waiting_satellite = satellite_num

            # self.traffic = 'OFF'
            # self.shut_down(constellation)

    def shut_down(self,constellation):
        if self.access != None: 
            constellation[self.access].channel_overhead -= 1
        elif self.access_flag == True:  
            satellite_num = self.candidate[self.action]
            assert satellite_num != -1, 'accessing the satellite -1'
            constellation[satellite_num].channel_overhead -= 1
        elif self.waiting_satellite != None:  
            constellation[self.waiting_satellite].queue.remove(self)  
        else:  
            print('Wrong in support.shutdown')
        self.waiting_satellite = None
        self.access = None
        self.action = None
#        self.candidate = [-1 for x in range(PS.candidate_num)]
#        self.elevation = [(0.0, 0.0, 0.0) for x in range(PS.candidate_num)]
#        self.condition = [-1 for x in range(PS.candidate_num)]
        self.old_state = None
        self.access_flag = None
        self.access_timer = None
        self.cost = 0.0
        self.reward = []
        self.packets_now = None


class Satellite(object):
    def __init__(self):
        self.index = None
        self.position = [0.0,0.0]       #satellie.position = (x,y)
        self.channel_overhead = 0
        self.queue = []  
    def move(self):
        self.position[1] += (PS.w * PS.slot)     


def gen_Constellation():
    inter_phi = PS.intre_phi
    intra_phi = PS.intra_phi
    random_position = (random.uniform(0,360),random.uniform(0,360))     
    X = [Satellite() for x in range(PS.satellite_per_plane  * PS.plane_num)]
    for index,satellite in enumerate(X):       
        satellite.index = index
    for x in range(PS.plane_num):
        for y in range(PS.satellite_per_plane):
            X[x * PS.satellite_per_plane + y].position = [random_position[0] + x * inter_phi,random_position[1] + y * intra_phi]
    return X


def gen_User_Set():
    X = [User() for x in range(PS.user_num)]
    return X

def gen_neighborhood(user_set):
    for user in user_set:
        user_neighborhood = []     
        user_distance = []        
        for y in user_set:
            distance = dp.cal_ground_distance(user,y)
            if (user != y) and (distance < PS.Rr):
                # 用户y是用户x的邻居，开始插入
                if len(user_neighborhood) == 0:   
                    assert len(user_distance) == 0, print('x_distance =',user_distance)
                    user_neighborhood.append(y)
                    user_distance.append(distance)
                else:
                    for index,x in enumerate(user_neighborhood):
                        if user_distance[index] > distance:
                            break
                    user_neighborhood.insert(index,y)
                    user_distance.insert(index,distance)
        user.neighborhood = user_neighborhood
        user.distance = user_distance

def gen_position(user_set):
    # assert math.isclose(PS.position_delta_deg,Position.position_delta_deg) , 'delta_deg not match！'
    X = Position.X
    Y = Position.Y
    for index,user in enumerate(user_set):
        user.position = np.array([X[index],Y[index]])


    # start_point = (PS.centre[0] - PS.delta_deg/2,PS.centre[1] - PS.delta_deg/2)
    # for x in range(PS.user_num_per_line):
    #     for y in range(PS.user_num_per_line):
    #         user_id = x * PS.user_num_per_line + y
    #         X_position = (start_point[0] + y * PS.delta_deg_per_user , start_point[0] + (y + 1) * PS.delta_deg_per_user)
    #         Y_position = (start_point[1] + x * PS.delta_deg_per_user , start_point[1] + (x + 1) * PS.delta_deg_per_user)
    #         user_set[user_id].position = np.array([random.uniform(X_position[0],X_position[1]),random.uniform(Y_position[0],Y_position[1])])       #经度[0],纬度[1]


def constellation_move(constellation):
    for x in constellation:
        x.move()

def system_handover(user_set,constellation,count):
    store_constellation = copy.deepcopy(constellation)
    for user in user_set:
        if user.traffic == 'ON':
            old_satellite = user.candidate[user.action]    
            if user.old_state != None and user.waiting_satellite is None:  #只有在前一切换帧进行了决策且不在等待阶段的用户才进行训练
                S = user.old_state
                A = user.action
                assert len(user.reward) == PS.T_H, print(user.reward)
                R = np.mean(user.reward)
                S_ = dp.state_output(user, store_constellation)
                dp.learn(S[A], A, R, S_[A],user.beta, user.network)  
            user.old_state = dp.state_output(user,store_constellation)
            user.action = dp.predict(user)

            satellite_num = user.candidate[user.action]
            user.reward = []
            if user.access is not None: 
                if user.access == satellite_num:  
                    pass  
                else:
                    constellation[old_satellite].channel_overhead -= 1  
                    constellation[satellite_num].queue.append(user) 
                    user.waiting_satellite = satellite_num 
                    user.access = None
                    user.cost = None
                    user.access_flag = False
                    user.access_timer = None
            else:  
                if satellite_num != old_satellite: 
                    if user.access_flag == True: 
                        constellation[old_satellite].channel_overhead -= 1  
                        constellation[satellite_num].queue.append(user) 
                        user.waiting_satellite = satellite_num  
                        user.cost = None
                        user.access_flag = False
                        user.access = None
                        user.access_timer = None
                    else:
                        assert user.waiting_satellite == old_satellite
                        user.waiting_satellite = satellite_num
                        constellation[old_satellite].queue.remove(user)  
                        constellation[satellite_num].queue.append(user)  
                else: 
                    if user.access_flag == True:  
                        count.sum_handovertimes += 1
                        user.cost = PS.Bc * dp.U_C(store_constellation[satellite_num].channel_overhead)
                        user.access_flag = True
                        user.access = None
                        user.access_timer = PS.T_h
                    else: 
                        assert user.waiting_satellite == old_satellite
                        assert user in constellation[user.waiting_satellite].queue, print('system handover：用户不在卫星队列里')

    for satellite_num, satellite in enumerate(constellation):
        while len(satellite.queue) > 0 :
            if satellite.channel_overhead < PS.C_max:
                count.sum_handovertimes += 1
                user = satellite.queue.pop(0)
                user.waiting_satellite = None
                satellite.channel_overhead += 1
                user.cost = PS.Bc * dp.U_C(store_constellation[satellite_num].channel_overhead)
                user.access = None
                user.access_flag = True
                user.access_timer = PS.T_h
            else:
                count.sum_waiting_times += len(satellite.queue)
                break

def system_normal(user_set,constellation,count):     
    waiting_user = 0 

    for user in user_set:
        if user.traffic == 'ON':
            if user.waiting_satellite is None:
                count.sum_cost += user.cost          
                assert user.action != None
                # active_user += 1  
              
                if user.access_flag == True:    
                    # reward_now += (-user.cost)
                    # user.reward.append(-user.cost)
                    user.packets_now = None    
                    user.access_timer -= 1
                    if user.access_timer == 0:     
                        SINR = dp.cal_SINR(user, user_set, constellation)      
                        R = dp.rate(SINR)                                      
                        if R < PS.R_min:       
                            count.sum_failtimes += 1
                            user.access_flag = True
                            user.access = None
                            user.access_timer = PS.T_h
                        else:                 
                            count.sum_successtimes += 1
                            user.access_flag = False
                            user.access = user.candidate[user.action]
                            user.access_timer = None

                else:                         
                    SINR = dp.cal_SINR(user, user_set, constellation)
                    R = dp.rate(SINR)
                    if R < PS.R_min:
                        user.packets_now = 0
                    else:
                        N_packets = int(R * PS.slot / PS.packet_size)
                        user.packets_now = N_packets       
                        # reward_now += (N_packets * PS.Br - user.cost)
                        count.sum_throughput += N_packets
            else:
                waiting_user += 1
    # average_reward = (reward_now / active_user if active_user != 0 else 0.0)     
    # count.sum_reward += average_reward
    system_reward = 0.0       
    active_users = 0
    for user in user_set:           
        if user.traffic == 'ON' and user.waiting_satellite is None:
            active_users += 1     
            if user.packets_now == None:        
                user.reward.append(-user.cost)
                system_reward += (-user.cost)     
            else:                              
                system_reward += (user.packets_now * PS.Br - user.cost)
                sum_packets = user.packets_now     
                sum_number = 1                      
                for neighbor in user.neighborhood:
                    if neighbor.packets_now != None:  
                        sum_packets += neighbor.packets_now
                        sum_number += 1
                average_packets = sum_packets / sum_number
                user.reward.append(average_packets * PS.Br - user.cost)

    count.sum_reward_2 += system_reward
    count.sum_active_user += active_users
    count.sum_waiting_user += waiting_user

    average_reward = (system_reward / active_users if active_users != 0 else 0.0)  
    count.sum_reward += average_reward