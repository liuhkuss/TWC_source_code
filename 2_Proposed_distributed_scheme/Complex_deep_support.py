import keras.initializers
import numpy as np
import Complex_Parameter_Set as PS
import math
import random
from scipy import special
from Replay import SumTree
# from collections import deque
import copy
import tensorflow as tf
# import Replay as rp
import time
# from scipy import integrate
# from pynverse import inversefunc


Re = PS.Re
height = PS.height
pi = PS.pi

def sind(x):
    return math.sin(x * pi / 180)

def cosd(x):
    return math.cos(x * pi / 180)

def tand(x):
    return math.tan(x * pi / 180)

def atand(x):
    return math.atan(x) / pi * 180

def acosd(x):
    return math.acos(x) / pi * 180


def random_pick(list,probabilitis):
#    sum = 0.0
#    for pro in probabilitis:
#        sum += pro
#    x = random.uniform(0,sum)
    x = random.uniform(0,1)
    cum_pro = 0.0
    for item,item_pro in zip(list,probabilitis):
        cum_pro += item_pro
        if x < cum_pro:
            break
    return item


def cal_distance(user,satellite):
    a = Re / (Re + height)
    x = user.position[0] - satellite.position[0]
    d = (Re + height) * math.sqrt(
        1 + pow(a,2) - 2 * a * (cosd(x) * cosd(user.position[1]) * cosd(satellite.position[1]) +
                                sind(user.position[1]) * sind(satellite.position[1])))
    return d

def cal_ground_distance(user1,user2):
    phi_1 = user1.position[1]       
    phi_2 = user2.position[1]       
    lambda_1 = user1.position[0]    
    lambda_2 = user2.position[0]   

    temp = min(1.0,sind(phi_1)*sind(phi_2) + cosd(phi_1)*cosd(phi_2)*cosd(lambda_1 - lambda_2))
    theta = math.acos(temp) 
    d = PS.Re * theta
    return d

def condition_trans(old_condition,theta,N_slot):     
    condition_list = [0,1,2]
    assert theta >= PS.theta_min, '候选集中卫星的仰角小于theta_min'
    if theta < PS.theta_min:
        new_condition = -1
    else:
        num = min(math.floor(theta/PS.delta_deg)-1,PS.model_num - 1)   
        if old_condition == -1:    
            assert (num == math.floor(PS.theta_min/PS.delta_deg)-1) or (N_slot == 0), '没有以最小角度进入'      
            new_condition = random_pick(condition_list,PS.static_pro[num])
        else:                     
            if N_slot % PS.T_F == 0:
                new_condition = random_pick(condition_list, PS.trans_pro[num][old_condition])
            else:
                new_condition = old_condition
    return new_condition

def cal_SINR(user,user_set,constellation):
    P_desired = 0.0    
    P_inf = 0.0        
    P_noise_dbm = PS.f_noise + 10 * math.log10(PS.B)
    P_noise = pow(10,P_noise_dbm/10) * 0.001   
    for x in user_set:
        if x.traffic == 'ON':
            satellite_num = x.candidate[x.action]  
            P = gen_power(user,x,constellation[satellite_num])   
            if x == user:
                P_desired += P
            else:
                P_inf += P
    SINR = P_desired / (P_inf + P_noise)
    return SINR


def cal_elevation(user,satellite):
    x = user.position[0] - satellite.position[0]
    num = cosd(x) * cosd(user.position[1])*cosd(satellite.position[1]) +\
          sind(user.position[1])*sind(satellite.position[1]) - Re/(Re + height)
    den = math.sqrt(1 - pow(cosd(x) * cosd(user.position[1]) * cosd(satellite.position[1]) +
                            sind(user.position[1]) * sind(satellite.position[1]) , 2))
    return (atand(num/den) if den > 0 else 90)


def cal_off_axis(satellite,user1,user2):
    x =  sind(user1.position[1]) * sind(user2.position[1]) +\
        cosd(user1.position[1]) * cosd(user2.position[1]) * cosd(user1.position[0] - user2.position[0])
    earth_angle = acosd(x)
    d_u1_u2 = 2 * PS.Re * sind(earth_angle/2)
    d_u1_s = cal_distance(satellite,user1)
    d_u2_s = cal_distance(satellite,user2)
    #cos_off_axis = (pow(d_u1_s,2) + pow(d_u2_s,2) - pow(d_u1_u2,2)) / (2 * d_u1_s * d_u2_s)
    off_axis = acosd((pow(d_u1_s, 2) + pow(d_u2_s, 2) - pow(d_u1_u2, 2)) / (2 * d_u1_s * d_u2_s))
    return off_axis
'''
    if cos_off_axis > 1:
        assert cos_off_axis < 1.01, print('cos_off_axis = ',cos_off_axis)
        off_axis = 0.0
    else:
        off_axis = acosd((pow(d_u1_s, 2) + pow(d_u2_s, 2) - pow(d_u1_u2, 2)) / (2 * d_u1_s * d_u2_s))
    return off_axis
'''


def gen_small_channel(user,satellite):
    small_channel = 0.0
    for index,x in enumerate(user.candidate):        
        if x == satellite.index:   
            condition = user.condition[index]       
            elevation = cal_elevation(user, satellite)      
            num = min(math.floor(elevation / PS.delta_deg) - 1, PS.model_num - 1)       
            small_channel = random_pick(PS.value,PS.value_pros[num][condition])         
            break
    return small_channel


'''
def gen_small_channel(satellite,user):
    elevation = cal_elevation(satellite,user)
    function_f = lambda a: f_hs_theta(elevation, a)
    function_F = lambda x: integrate.quad(function_f,0.0,x)[0]
    y_values = random.random()
    hs_2 = float(inversefunc(function_F,y_values,domain = [0,10],open_domain = True))
    return 10 * math.log10(hs_2)
'''

def gen_large_channel(user,dst_user,satellite):
    #G_max = PS.G_max
    #phi_I = PS.phi_I
    #phi_3db = PS.phi_3db
    f_I = PS.f_I
    c = 3e+8

    d_I = cal_distance(user,satellite)
    L_I = pow((c / (4 * pi * f_I * d_I * 1000)), 2)  
    #L_I = 10 * math.log10(L_I)  

    d_I_dst = cal_distance(dst_user,satellite)
    L_I_dst = pow((c / (4 * pi * f_I * d_I_dst * 1000)), 2)
    G_max = PS.G_max_K / L_I_dst            
    #u_I = 2.07123 * sind(phi_I) / sind(phi_3db)
    u_I = 2.07123 * cal_ground_distance(user,dst_user) / PS.R_3dB
    b_I = (1.0 if u_I == 0 else pow((special.jv(1, u_I) / (2 * u_I) + 36 * special.jv(3, u_I) / pow(u_I, 3)), 2) )
    G_S = G_max * b_I

    G_I = PS.G_I
    return G_S * G_I * L_I

def gen_channel(user,dst_user,satellite):
    large_channel = gen_large_channel(user,dst_user,satellite)
    small_channel = gen_small_channel(user,satellite)
    return large_channel * small_channel

def gen_power(user,dst_user,satellite):     
    P_K_W = pow(10,PS.P_K / 10)  
    channel = gen_channel(user,dst_user,satellite)
    return P_K_W * channel

def rate(SINR):
    return PS.B * math.log2(1 + SINR)

def U_C(x):
    if not (x >= 0 and x <= PS.C_max):
        print(x)        
    C_max = PS.C_max
    K = PS.K
    R_min = 1.0
    R_max = C_max
    a = 10.0 / (R_max + R_min)
    b = (R_max + R_min) / 2.0
    y = (K-1.0)/(1.0 + math.exp(-a * (x - b)))
    return y + 1.0


def state_output(user,constellation):      
    state = []
    for index,x in enumerate(user.candidate):
        if x == -1:    
            state.append(-1)
        else:
            satellite_state = []
            theta = user.elevation[index][0] / 90.0  
            if user.elevation[index][1] != 0.0:
                theta_v = (user.elevation[index][0] - user.elevation[index][1])/90.0      
                if user.elevation[index][2] != 0.0:
                    theta_a = (user.elevation[index][0] + user.elevation[index][2] - 2*user.elevation[index][1])/90.0
                else:
                    theta_a = 0.0
            else:
                theta_v = 0.0
                theta_a = 0.0
            satellite_state.extend([theta,theta_v,theta_a])
            binary_condition = [0.0,0.0,0.0]
            condition = user.condition[index]
            assert condition != -1, '候选集中出现不可视卫星'
            binary_condition[condition] = 1.0
            satellite_state.extend(binary_condition)
            if x == user.access:
                cost = user.cost/(PS.K * PS.Bc)                          
            else:
                cost = (PS.Bc * U_C(constellation[x].channel_overhead))/(PS.K * PS.Bc)
            assert cost <= 1.0,'Cost is out of range!'
            satellite_state.append(cost)
            access = (1.0 if user.access == x else 0.0)
            satellite_state.append(access)
            # satellite_state = [theta,theta_v,theta_a,binary_condition,cost,access]

            # neighborhood = copy.deepcopy(user.neighborhood)
            # distance = [cal_ground_distance(user,x) for x in user.neighborhood]
            # for i in range(PS.neighborhood_num):
            #     if len(neighborhood) == 0:
            #         d = 1.0
            #         theta = 0.0
            #         theta_v = 0.0
            #         theta_a = 0.0
            #         binary_condition = [0.0,0.0,0.0]
            #         satellite_state.extend([d,theta,theta_v,theta_a])
            #         satellite_state.extend(binary_condition)
            #     else:
            #         neighbor_index = distance.index(min(distance))   
            #         neihgbor = neighborhood[neighbor_index]             
            #
            #         
            #         d = distance[neighbor_index] / PS.Rr
            #
            #        
            #         index = -1
            #         for candidate_index,candidate_satellite in enumerate(neihgbor.candidate):
            #             if x == candidate_satellite:
            #                 index = candidate_index
            #                 break
            #         #index = neihgbor.candidate.index(x)    
            #
            #         if index != -1:     
            #             theta = neihgbor.elevation[index][0] / 90.0  
            #             if neihgbor.elevation[index][1] != 0.0:
            #                 theta_v = (neihgbor.elevation[index][0] - neihgbor.elevation[index][1]) / 90.0  
            #                 if neihgbor.elevation[index][2] != 0.0:
            #                     theta_a = (neihgbor.elevation[index][0] + neihgbor.elevation[index][2] - 2 * user.elevation[index][
            #                         1]) / 90.0
            #                 else:
            #                     theta_a = 0.0
            #             else:
            #                 theta_v = 0.0
            #                 theta_a = 0.0
            #
            #             binary_condition = [0.0, 0.0, 0.0]
            #             condition = neihgbor.condition[index]
            #             assert condition != -1, '候选集中出现不可视卫星'
            #             binary_condition[condition] = 1.0
            #
            #         else:
            #             theta = 0.0
            #             theta_v = 0.0
            #             theta_a = 0.0
            #             binary_condition = [0.0, 0.0, 0.0]
            #
            #         satellite_state.extend([d, theta, theta_v, theta_a])
            #         satellite_state.extend(binary_condition)
            #
            #         del(neighborhood[neighbor_index])
            #         del(distance[neighbor_index])
            # satellite_state.append(user.beta)       
            state.append(satellite_state)
    return state


def predict(user):
    if random.random() < user.epsilon:
        X = []
        for index,x in enumerate(user.candidate):
            if (x != -1) and (user.elevation[index][0] > PS.theta_access):    
                X.append(index)
        if X == []:
            print('Wrong!')
        action = random.choice(X)
    else:
        X = []
        for index,state in enumerate(user.old_state):
            x = user.candidate[index]          
            if (x != -1) and (user.elevation[index][0] > PS.theta_access) : 
                state = np.array(state)  
                X.append(user.network.evaluate_model(np.expand_dims(state, axis=0)).numpy()[0][0])
            else:
                X.append(-float('inf'))
            # if (state[0] != 0.0) or (state[1] != 1.0):      
            #     X.append(-float('inf'))
            # else:
            #     state = np.array(state)     
            #     X.append(user.network.evaluate_model(np.expand_dims(state, axis=0)).numpy()[0][0])
        action = X.index(max(X))
    return action


def learn(old_state,action,reward,new_state,network):
    reward = reward / 10.0  
    priority = 1.0          
#    priority = old_state[0]     
    down = 1.0 if new_state[0] == 0.0 else 0.0
    network.replay_pool.add(priority,(old_state,action,reward,new_state,down))
    #if len(network.replay_pool) >= PS.batch_size:
    if network.replay_pool.full == 1:
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_down = []
        for x in range(PS.batch_size):
            index,get_priority,data = network.replay_pool.get_leaf()
            batch_state.append(data[0])
            batch_action.append(data[1])
            batch_reward.append(data[2])
            batch_next_state.append(data[3])
            batch_down.append(data[4])
        #, batch_action, batch_reward, batch_next_state = \
        #    map(np.array, zip(*random.sample(network.replay_pool, PS.batch_size)))
        (batch_state,batch_action,batch_reward,batch_next_state,batch_down) =\
            map(np.array,(batch_state,batch_action,batch_reward,batch_next_state,batch_down))
        q_value = network.target_model(batch_next_state)
        y = batch_reward + (PS.gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_down)  

        with tf.GradientTape() as tape:
            loss = tf.keras.losses.mean_squared_error(  
                y_true=y,
                y_pred=tf.reduce_max(network.evaluate_model(batch_state), axis=1)
            )
        grads = tape.gradient(loss, network.evaluate_model.variables)
        network.optimizer.apply_gradients(grads_and_vars=zip(grads, network.evaluate_model.variables))  

        network.timer -= 1
        #print(loss.numpy())
        if network.timer == 0:
            # print('loss = ',loss)
            #print('q_value = ',q_value)
            network.target_model = copy.deepcopy(network.evaluate_model)
            network.timer_reset()



class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  
        self.dense1 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu,
                                            kernel_initializer=keras.initializers.GlorotUniform(seed=0))
        self.dense2 = tf.keras.layers.Dense(units=150, activation=tf.nn.relu,
                                            kernel_initializer=keras.initializers.GlorotUniform(seed=0))
        self.dense3 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu,
                                            kernel_initializer=keras.initializers.GlorotUniform(seed=0))
        self.dense4 = tf.keras.layers.Dense(units=1)     

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)


class network(object):
    def __init__(self):
        self.evaluate_model = QNetwork()
        self.target_model = copy.deepcopy(self.evaluate_model)
        self.replay_pool = SumTree(capacity=PS.Replay_capacity)
        #self.replay_pool = deque(maxlen=1024)
        #self.replay_pool = rp.Memory(capacity=1024)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=PS.learning_rate)        
        self.timer_reset()                   

    def timer_reset(self):
        self.timer = 100
