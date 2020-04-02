#!/usr/bin/env python
# coding: utf-8

# In[40]:


import torch
import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():
    #torch.cuda.set_device(0) 
    device='cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #device=torch.cuda.current_device()
else:
    device='cpu'
    torch.set_default_tensor_type('torch.FloatTensor')
    
class Heston(nn.Module):
    
    def __init__(self, timegrid, strikes_call, strikes_put, device):
        
        super(Heston, self).__init__()
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.strikes_put = strikes_put
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)

        
    def forward(self, S0, V0, rate, indices, z,z1, MC_samples): 
        S_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        V_old_abs = torch.repeat_interleave(V0, MC_samples, dim=0)    
        K_call = self.strikes_call
        K_put = self.strikes_put
        zeros = torch.repeat_interleave(torch.zeros(1,1), MC_samples, dim=0)
        average_SS = torch.Tensor()
        average_SS1 = torch.Tensor()
        average_SS_OTM = torch.Tensor()
        average_SS1_ITM = torch.Tensor()
        # use fixed step size
        h = self.timegrid[1]-self.timegrid[0]
        n_steps = len(self.timegrid)-1
        # set maturity counter
        countmat=-1
        
        # Solve for S_t, V_t (Euler)
        
        for i in range(1, len(self.timegrid)):
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(MC_samples,1)
            S_new = S_old+S_old*rate*h+S_old*torch.sqrt(V_old_abs)*dW1
            S_old = S_new
            V_new = V_old_abs + 1.5*(0.04-V_old_abs)*h+0.3*torch.sqrt(V_old_abs)*dW
            V_old_abs = torch.cat([V_new,zeros],1)
            V_old_abs= torch.max(V_old_abs,1,keepdim=True)[0]
        
            # If particular timestep is a maturity for Vanilla option
            
            if int(i) in indices:
                countmat+=1
                Z_new=torch.Tensor()
                Z_newP_ITM = torch.Tensor()
                Z_newP_OTM = torch.Tensor()
                countstrikecall=-1
                
            # Evaluate put (OTM) and call (OTM) option prices 
                
                for strike in K_call:
                    countstrikecall+=1
                    strike = torch.ones(1,1)*strike
                    strike_put = torch.ones(1,1)*K_put[countstrikecall]
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    K_extended_put = torch.repeat_interleave(strike_put, MC_samples, dim=0).float()

                    # Since we use the same number of maturities for vanilla calls and puts: 
                    
                    price = torch.cat([S_old-K_extended,zeros],1) #call OTM
                    price_OTM = torch.cat([K_extended_put-S_old,zeros],1) #put OTM
                    
                    # Discounting assumes we use 2-year time horizon 
                    
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    price_OTM = torch.max(price_OTM, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    
                
                    Z_new= torch.cat([Z_new,price],1)  
                    Z_newP_OTM= torch.cat([Z_newP_OTM,price_OTM],1)  
                    
               # MC step:
            
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)
                avg_SSP_OTM = torch.cat([p.mean().view(1,1) for p in Z_newP_OTM.T], 0)
                average_SS = torch.cat([average_SS,avg_S.T],0) #call OTM
                average_SS_OTM = torch.cat([average_SS_OTM,avg_SSP_OTM.T],0) #put OTM       
                countstrikeput=-1
                
          # Evaluate put (ITM) and call (ITM) option prices 
                
                Z_new=torch.Tensor()
                for strike in K_put:
                    countstrikeput+=1
                    strike = torch.ones(1,1)*strike
                    strike_call = torch.ones(1,1)*K_call[countstrikeput]
                    K_extended = torch.repeat_interleave(strike, MC_samples, dim=0).float()
                    K_extended_call = torch.repeat_interleave(strike_call, MC_samples, dim=0).float()
                    price_ITM = torch.cat([K_extended_call-S_old,zeros],1) #put ITM
                    price = torch.cat([S_old-K_extended,zeros],1) #Call ITM
                    price = torch.max(price, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    price_ITM = torch.max(price_ITM, 1, keepdim=True)[0]*torch.exp(-rate*1*i/n_steps)
                    
                    
                    Z_new= torch.cat([Z_new,price],1) 
                    Z_newP_ITM= torch.cat([Z_newP_ITM,price_ITM],1)    
                    
            # MC step         
                    
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)
                avg_SSP_ITM = torch.cat([p.mean().view(1,1) for p in Z_newP_ITM.T], 0)
                average_SS1_ITM = torch.cat([average_SS1_ITM,avg_SSP_ITM.T],0)                            
                average_SS1 = torch.cat([average_SS1,avg_S.T],0)      
                    
            # Return model implied vanilla option prices    
                
        return torch.cat([average_SS,average_SS_OTM,average_SS1,average_SS1_ITM  ],0)  
    
MC_samples=500000
n_steps=360
rho = -0.9
iter_count = 11
Z2=np.zeros((48,46))
strikes_put=np.arange(55, 101, 1).tolist()
strikes_call=np.arange(100, 146, 1).tolist()
n_steps=360
timegrid = torch.linspace(0,1,n_steps+1) 
indices = torch.tensor([30, 60, 90, 120, 150,180 , 210, 240, 270, 300, 330, 360])  
model = Heston(timegrid=timegrid, strikes_call=strikes_call,strikes_put=strikes_put, device=device)
S0 = torch.ones(1, 1)*100
V0 = torch.ones(1,1)*0.04
rate = torch.ones(1, 1)*0.025

for i in range(1,iter_count):
    np.random.seed(i)
    z_1 = np.random.normal(size=(MC_samples, n_steps))
    z_2 = np.random.normal(size=(MC_samples, n_steps))
    z_1 = np.append(z_1,-z_1,axis=0)
    z_2 = np.append(z_2,-z_2,axis=0)
    zz  = rho*z_1+np.sqrt(1-rho ** 2)*z_2
    z_1 = torch.tensor(z_1).to(device=device).float()
    z_2 = torch.tensor(z_2).to(device=device).float()
    zz = torch.tensor(zz).to(device=device).float()
    print('current batch of milion samples:', i)
    model=model.to(device=device)
    Z=model(S0, V0, rate, indices, z_1,zz, 2*MC_samples).float().to(device=device)
    Z=Z.detach().to(device='cpu').numpy()/(iter_count-1)
    Z2=Z2+Z 
    
Call_OTM_Unit=Z2[0:12,:]
Put_OTM_Unit=Z2[12:24,:]
Call_ITM_Unit=Z2[24:36,:]
Put_ITM_Unit=Z2[36:48,:]

torch.save(Call_OTM_Unit,'Call_OTM_Unit.pt')
torch.save(Put_OTM_Unit,'Put_OTM_Unit.pt')
torch.save(Call_ITM_Unit,'Call_ITM_Unit.pt')
torch.save(Put_ITM_Unit,'Put_ITM_Unit.pt')    
    


# In[ ]:




