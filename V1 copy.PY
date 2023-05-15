import numpy as np
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
# 单GPU或者CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#RANDME--------------------加大训练时长， 消除噪音0.5

batch_size = 8
Mr = 64
Time_Frame_ = 10
iteration = 100
batch_per_epoch = 200
learning_rate = 0.0001
agent_num = 2
channel_number = 3
state_number = channel_number+1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers =  [512,1024,1024,1024,channel_number]   
        self.net1 = nn.Sequential(
            nn.Linear(layers[0],layers[1]),
            nn.ELU(),
            nn.Linear(layers[1],layers[2]),
            nn.ELU(),
            nn.Linear(layers[2],layers[3]),
            nn.ELU(),
            nn.Linear(layers[3],layers[4]),   
            
            )
        self.net2 = nn.Sequential(
            nn.Linear(layers[0],layers[1]),
            nn.ELU(),
            nn.Linear(layers[1],layers[2]),
            nn.ELU(),
            nn.Linear(layers[2],layers[3]),
            nn.ELU(),
            nn.Linear(layers[3],layers[4]),
            nn.Sigmoid()
            )
        self.rnn = nn.LSTMCell(channel_number,512)
        self.rnn = self.rnn.to(device)
    
    def enviroment(self,w):
        real_w0 = w[:,0:Mr]
        imag_w0 = w[:,Mr:2*Mr]
        
        w_complex = torch.complex(real_w0,imag_w0)
        return w_complex



    def forward(self,x_data,h_,C_):
        
        ###############前面的unit
        
        for j in range(Time_Frame_):
            h_ , C_ =self.rnn(x_data , (h_ , C_))
            w = self.net1(h_)   #(8 x 3)

            

            
            # w = Net.enviroment(self,w).to(device)                                            #(batch_size x 64)
            # wt = torch.transpose(w,dim0 =0,dim1 = 1).to(device)                              # (64 X batch_size)
            
            # y_0 = yt(wt).to(device)
            x_data = torch.as_tensor(w,dtype=torch.float32) 
            x_data = x_data.to(device)   


        ##########################最后一个LSTM
        h_ ,C_ = self.rnn(x_data, (h_, C_))
        # h_ = h_.to(device)
        # C_ = C_.to(device)

        
        #########################最后一个DNNk       
        v = self.net2(C_)
        
        # v_ = v_complex  
        v_ = torch.round(v)                            
        # v_ = v             #[batch_size , 3]

        return v_
        







def loss_function(k):

   
    aj = [0,0,0]
    rewards = []
    for i in range (batch_size):
        aj0 = k[0][i][0]+k[1][i][0]
        aj1 = k[0][i][1]+k[1][i][1]
        aj2 = k[0][i][2]+k[1][i][2]
    
        aj0  = aj0.item()
        aj1  = aj1.item() 
        aj2  = aj2.item()
        aj3  = aj1 + aj0 + aj2
        if aj3 != 2:
            reward = -99
        else:
            if  aj0 == 2 :                     # 2 idle
                reward = 0
            if  aj1 == 2 or aj2 == 2 :       # collision
                reward = -5
            if  aj1 ==1 and aj2 == 1:        # sucsess
                reward = 5
            if aj1 == 1 and  aj0 == 1:       # 1idle,1sucsess
                reward = 2 
            if aj2 == 1 and  aj0 == 1:
                reward = 2
            
        rewards.append(reward)
    rewards = torch.tensor(rewards,dtype = float,requires_grad=True)
    # ab2 = torch.sum(rewards)
    ab2 = torch.mean(rewards)
    Loss = -ab2
        

    return Loss 



#---------激活网络------------
agents = []
for i in range (agent_num):
    agent = Net()
    agent = agent.to(device)
    agents.append(agent)



# net = Net() 
# net = net.to(device)
print("if on cuda:",next(agents[0].parameters()).is_cuda)


#---------optimizer------------
# for agent in agents:
#     optimizer = torch.optim.Adam(agent.parameters(),lr=learning_rate)


optimizer1 = torch.optim.Adam(agents[0].parameters(),lr=learning_rate)
optimizer2 = torch.optim.Adam(agents[1].parameters(),lr=learning_rate)



#——————————trainning----------

plt.ion()
out2 = []
bf_opt_list = []
w = []
k = []
for i in range(iteration):
    
    for n_item in range(batch_per_epoch):
        

        # #---------得到信道------------
        # channel = initialize()
       
       
        #---------初始化------------
        C_ = torch.zeros([batch_size,512])
        h_ = torch.zeros([batch_size,512])
        y_0 =torch.randn([batch_size,channel_number])
        C_ = C_.to(device) 
        h_ = h_.to(device)
        y_0 = y_0.to(device)


        

 
        # 前向传播           接入y_0 , h_ , C_
        for j in range(2):
            k_  = agents[j].forward(y_0,h_,C_)
            k.append(k_)    
            
        #---------计算loss------------
        loss1  = loss_function(k)
        loss1 = loss1.to(device)

        loss2  = loss_function(k)
        loss2 = loss2.to(device)
        
                       
        #---------反向传播------------
        optimizer1.zero_grad()  ## reset gradient
        
        loss1.backward()  ## back propagation
        optimizer1.step()  ## update parameters of net

        optimizer2.zero_grad()  ## reset gradient
        loss2.backward()  ## back propagation
        optimizer2.step()  ## update parameters of net
        print(loss1)


        


    out2.append(loss1.item())
    
    
    plt.plot(out2)
    

    # plt.plot(bf_opt_list, label='MRT: {}'.format(np.mean(bf_opt)))

    plt.xlabel('epoch')
    plt.ylabel('beamforming gain (dB)')
    plt.xlim(-10, iteration)
    plt.grid()
    plt.legend()   


    plt.savefig('fig/Time_Frame_{}_gain2.png'.format(Time_Frame_), dpi=300)   
    plt.pause(1) 
    plt.clf()


plt.ioff()    
plt.show()


#### save
# torch.save(net,'Model/Model'+'_'+str(int(Time_Frame_))+'_'+'Time_Frames2.pth')
# np.save('line/Time_Frame_{}_gain.npy'.format(Time_Frame_),out2)
