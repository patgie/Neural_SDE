import torch
import torch.nn as nn
import numpy as np
import math
import os
import time
from random import randrange



if torch.cuda.is_available():
    device = 'cuda'
    # Uncomment below to pick particular device if running on a cluster:
    torch.cuda.set_device(3) 
    device='cuda:3'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device=torch.cuda.current_device()

else:
    device='cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

class Net_timestep(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation = "relu"):
        super(Net_timestep, self).__init__()
        self.dim = dim
        self.nOut = nOut
        
        if activation!="relu" and activation!="tanh":
            raise ValueError("unknown activation function {}".format(activation))
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)
        
    def hiddenLayerT0(self,  nIn, nOut):
        layer = nn.Sequential(#nn.BatchNorm1d(nIn, momentum=0.1),
                              nn.Linear(nIn,nOut,bias=True),
                              #nn.BatchNorm1d(nOut, momentum=0.1),   
                              self.activation)   
        return layer
    
    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                              #nn.BatchNorm1d(nOut, momentum=0.1),  
                              self.activation)   
        return layer
    
    
    def outputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut,bias=False))
        return layer
    
    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output
    
#Set up neural SDE class    

class Net_SDE(nn.Module):
    
    def __init__(self, dim, timegrid, strikes_call, strikes_put, ITM_call, OTM_call, ITM_put, OTM_put, n_layers, vNetWidth, device):
        
        super(Net_SDE, self).__init__()
        self.dim = dim
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        self.strikes_put = strikes_put
        
        #Input to each coefficient (NN) will be (t,S_t,V_t)
        
        self.diffusion =Net_timestep(dim=dim+2, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.driftV = Net_timestep(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV = Net_timestep(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        self.diffusionV1 = Net_timestep(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
        
    def forward(self, S0, V0, rate,BS_vol, indices, z,z1, MC_samples): 
        S_old = torch.repeat_interleave(S0, MC_samples, dim=0)
      # Uncomment when using BS Control Variate:  
      # BS_old = torch.repeat_interleave(S0, MC_samples, dim=0)
        V_old = torch.repeat_interleave(V0, MC_samples, dim=0)  
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
        irand = [randrange(0,n_steps+1,1) for k in range(48)]
        for i in range(1, len(self.timegrid)):
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            dW1 = (torch.sqrt(h) * z1[:,i-1]).reshape(MC_samples,1)
            current_time = torch.ones(1, 1)*self.timegrid[i-1]
            input_time  = torch.repeat_interleave(current_time, MC_samples,dim=0)
            inputNN = torch.cat([input_time.reshape(MC_samples,1),S_old, V_old],1)
            inputNNvol = torch.cat([input_time.reshape(MC_samples,1),V_old],1)
       
            if int(i) in irand:
                 S_new =S_old + S_old*rate*h + self.diffusion(inputNN)*dW 
                 V_new = V_old + self.driftV(inputNNvol)*h +self.diffusionV(inputNNvol)*dW + self.diffusionV1(inputNNvol)*dW1
            else:
                 S_new =S_old + S_old*rate*h + self.diffusion(inputNN).detach()*dW 
                 V_new = V_old + self.driftV(inputNNvol).detach()*h +self.diffusionV(inputNNvol).detach()*dW + self.diffusionV1(inputNNvol).detach()*dW1          
            S_new = torch.cat([S_new,zeros],1)
            S_new=torch.max(S_new,1,keepdim=True)[0]
            S_old = S_new
            V_old = V_new
        
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
                                    
            # Comment out Z_new and Z_newP_ITM if using CV with Black-Scholes 
                    
                    Z_new= torch.cat([Z_new,price],1) 
                    Z_newP_ITM= torch.cat([Z_newP_ITM,price_ITM],1)    
                    
            # MC step         
                    
                avg_S = torch.cat([p.mean().view(1,1) for p in Z_new.T], 0)
                avg_SSP_ITM = torch.cat([p.mean().view(1,1) for p in Z_newP_ITM.T], 0)
                average_SS1_ITM = torch.cat([average_SS1_ITM,avg_SSP_ITM.T],0)                            
                average_SS1 = torch.cat([average_SS1,avg_S.T],0)      
                    
            # Return model implied vanilla option prices    
                
        return torch.cat([average_SS,average_SS_OTM,average_SS1,average_SS1_ITM  ],0)  
    

def train_models(seedused):
    
    loss_fn = nn.MSELoss() 
    seedused=seedused+1
    torch.manual_seed(seedused)
    np.random.seed(seedused)
    model = Net_SDE(dim=1, timegrid=timegrid, strikes_call=strikes_call,strikes_put=strikes_put,ITM_call=ITM_call,OTM_call=OTM_call,ITM_put=ITM_put, OTM_put=OTM_put, n_layers=2, vNetWidth=20, device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, eps=1e-08,amsgrad=False,betas=(0.9, 0.999), weight_decay=0 )
  # optimizer= torch.optim.Rprop(model.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-07, 1))
    n_epochs = 500
    itercount = 0
    loss_val_best = 10
    
    
    for epoch in range(n_epochs):
        MC_samples_gen=200000
        n_steps=360
        path = "Neural_SDE_42"+".pth"
        torch.save(model.state_dict(), path)
        if epoch==250:
             optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, eps=1e-08,amsgrad=False,betas=(0.9, 0.999), weight_decay=0 )

        print('epoch:', epoch)
        
        
#evaluate and print RMSE validation error at the start of each epoch
        optimizer.zero_grad()
        pred = model(S0, V0, rate,BS_vol, indices, z_1,z_2, 2*MC_samples_gen).detach()
        loss_val=torch.sqrt(loss_fn(pred, target))
        print('validation {}, loss={}'.format(itercount, loss_val.item()))
        
        if loss_val < loss_val_best:
             loss_val_best=loss_val
             print('loss_val_best', loss_val_best)
             path = "Neural_SDE_49"+".pth"
             torch.save(model.state_dict(), path)

#store the erorr value

        losses_val.append(loss_val.clone().detach())
        batch_size = 20000
        permutation = torch.randperm(int(2*MC_samples_gen))
   
        for i in range(0,2*MC_samples_gen, batch_size):
            indices2 = permutation[i:i+batch_size]
            batch_x=z_1[indices2,:]
            batch_y=z_2[indices2,:]         
            timestart=time.time()
            optimizer.zero_grad()
            pred = model(S0, V0, rate,BS_vol, indices, batch_x,batch_y,batch_size)
            loss=torch.sqrt(loss_fn(pred, target))
            losses.append(loss.clone().detach())
            itercount += 1
            loss.backward()
            optimizer.step()
            print('iteration {}, loss={}'.format(itercount, loss.item()))
            print('time', time.time()-timestart)
            
    return seedused, model   

# Load market prices and set training target
ITM_call=torch.load('ITM_call_MC.pt').to(device=device)
ITM_put=torch.load('ITM_put_MC.pt').to(device=device)
OTM_call=torch.load('OTM_call_MC.pt').to(device=device)
OTM_put=torch.load('OTM_put_MC.pt').to(device=device)
target=torch.cat([OTM_call, OTM_put, ITM_call, ITM_put],0).float()


# Set up training
losses=[]
losses_val=[]
strikes_put=[55,60, 65,70,75,80,85,90,95,100]
strikes_call=[100,105,110,115,120,125,130,135, 140,145]
seedsused=np.zeros((101,1))
seedsused[0,0]=-1
S0 = torch.ones(1, 1)*100
BS_vol=torch.ones(1,1)*0.2
V0 = torch.ones(1,1)*0.04
rate = torch.ones(1, 1)*0.025

n_steps=360
# generate subdivisions of 2 year interval
timegrid = torch.linspace(0,1,n_steps+1) 
indices = torch.tensor([30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330, 360]) 

# Start training 100 models
# fix the seeds for reproducibility
np.random.seed(1)
MC_samples_gen=200000
z_1 = np.random.normal(size=(MC_samples_gen, n_steps))
z_2 = np.random.normal(size=(MC_samples_gen, n_steps))

# generate antithetics and pass to torch
z_1 = np.append(z_1,-z_1,axis=0)
z_2 = np.append(z_2,-z_2,axis=0)
z_1 = torch.tensor(z_1).to(device=device).float()
z_2 = torch.tensor(z_2).to(device=device).float()

for i in range(2,102):
    np.save('seeds_used_49.npy', seedsused) 
    np.save('losses_49.npy', losses)
    np.save('losses_49.npy', losses_val)
    seedsused[i-1,0],model=train_models(int(seedsused[i-2,0]))
    path = "Neural_SDE_49_"+str(i-1)+".pth"
    torch.save(model.state_dict(), path)  


