import math
import numpy as np
from scipy import integrate

import scipy.special

pi = 3.1415926
Re = 6400.0              
candidate_num = 30      
priority_num = 1        
T_H = 100               
T_F = 300               
R_3dB = 15          
Rr = 2.5 * R_3dB   
slot = 0.01        
T_h = 10              
T = 5.0*60.0              
Br = 1.0
Bc = 5.0
packet_size = 1000.0      
num_episode = 60       
num_exploration_episodes = 10  
initial_epsilon = 1.0           
final_epsilon = 0.0            
# initial_beta = 0.2
# final_beta = 0.8
learning_rate = 5e-6           
height = 1200.0          
theta_min = 20.0         
theta_access = 25.0       
we = 7.292 * (1e-5) / pi * 180.0      
w = math.sqrt(3.986e+5/pow((Re + height),3)) / pi * 180.0     
plane_num = 18                      
satellite_per_plane = 40            
intre_phi = 180.0/plane_num          
intra_phi = 360.0/satellite_per_plane 
user_num = 50
# user_num = user_num_per_line * user_num_per_line                     
C_max = 25                       
satellite_num = plane_num * satellite_per_plane     
# inclined = 87.9
# pro_fail = 0.1                      
# G_max = 30.0  
G_max_K = pow(10,-150/10)      
G_I = pow(10,0/10) 
f_I = 20e+9  
phi_I = 0.01    
phi_3db = 0.4   
P_K = 16.0 
B = 2e+6  
f_noise = -173 
Replay_capacity = 128           
batch_size = 32                
gamma = 0.7
K = 5.0

# length = (300,250,200,150,100,50)
length = 300.0
position_delta_deg = length / Re / pi * 180
centre = (116.0,40.0)              


# X_range = (116.0 - delta_deg/2,116.0 + delta_deg /2)
# Y_range = (40.0 - delta_deg/2,40.0 + delta_deg /2)
S = length * length            
S_r = pi * Rr * Rr       
rho = user_num / S       
neighborhood_num = math.ceil(rho * S_r)    


mean = (116,40)
cov = [[3.0,0],[0,3.0]]

R_min = 2e+6   
#SINR_tr = 0    

lamda = 0.1    
T_m = 180      
# a = 1.5    
# x_min = 60.  
# beta = 300.0    

delta_deg = 20.0
model_num = 4
static_pro = ((0.4000,0.2667,0.3333),
              (0.4546,0.3636,0.1818),
              (0.4666,0.2667,0.2667),
              (0.5000,0.2000,0.3000))
trans_pro_0 = ((0.8626,0.0737,0.0635),
               (0.1247,0.8214,0.0539),
               (0.0648,0.0526,0.8806))
trans_pro_1 = ((0.8681,0.0952,0.0367),
               (0.1300,0.8429,0.0271),
               (0.0701,0.0761,0.8538))
trans_pro_2 = ((0.8763,0.0724,0.0513),
               (0.1382,0.8201,0.0417),
               (0.0783,0.0533,0.8684))
trans_pro_3 = ((0.8870,0.0562,0.0568),
               (0.1489,0.8039,0.0472),
               (0.0890,0.0371,0.8739))
trans_pro = (trans_pro_0,trans_pro_1,trans_pro_2,trans_pro_3)

Loo_parameter_0 = ((-0.3,0.73,-15.9),(-8.0,4.5,-19.2),(-24.4,4.5,-19.0))
Loo_parameter_1 = ((-0.35,0.26,-16.0),(-6.3,1.4,-13.0),(-15.2,5.0,-24.8))
Loo_parameter_2 = ((-0.5,1.0,-19.0),(-5.6,1.2,-10.0),(-12.3,4.1,-16.0))
Loo_parameter_3 = ((-0.25,0.87,-21.7),(-6.6,2.3,-13.0),(-11.0,8.75,-24.2))
Loo_parameter = (Loo_parameter_0,Loo_parameter_1,Loo_parameter_2,Loo_parameter_3)

def Loo_pro_power(r,alpha,phi,MP):
    mu = math.log(10) * alpha / 20
    d_0 = pow(phi / 20 * math.log(10),2)
    b_0 = pow(10,MP / 10) / 2

    def f(z):
        y1 = 1/z
        y2 = math.exp(-pow(math.log(z) - mu ,2) / (2 * d_0) - (r + pow(z,2)) / (2 * b_0))
        y3 = scipy.special.i0(math.sqrt(r) * z / b_0)
        if y3 == float('inf'):
            y3 = 0
        return y1 * y2 * y3

    y1 = 1 / (2 * b_0 * math.sqrt(2 * pi * d_0))
    y2 = integrate.quad(f,0,10)[0]
    return y1 * y2


Step = 0.001
value = np.arange(0,5,Step)
value_pros = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

for i,parameter_i in enumerate(Loo_parameter):
    # Loo_parameter_i = ((-0.3,0.73,-15.9),(-8.0,4.5,-19.2),(-24.4,4.5,-19.0))
    for j,parameter_i_j in enumerate(parameter_i):
        #Loo_parameter_i_j = (-0.3,0.73,-15.9)
        Y = []
        CDF = 0.0
        for x in value:
            y = Loo_pro_power(x,parameter_i_j[0],parameter_i_j[1],parameter_i_j[2]) * Step
            Y.append(y)
            CDF += y
        print(CDF)
        value_pros[i][j] = [sample/CDF for sample in Y]