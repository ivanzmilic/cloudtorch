import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import sys
from astropy.io import fits
import cloud_model_torch as cmt 

# ====================================================================
def optimization(optimizer,niterations,parameters,model, xl, img):
    """
    This function performs the optimization of the parameters of the model.
    """
    from tqdm import trange
    t = trange(niterations, leave=True)
    for loop in t:
        optimizer.zero_grad()        #reset gradients
        parameters, final_output = evaluate(parameters, xl, img, model)

        chi2loss = torch.mean(torch.abs(img - final_output))
        #reguloss = chi2loss*0.0
        #reguloss += 1e-2*regu2(parameters, 0,img)
        #reguloss += 1e2*regu2(parameters, 1,img) 
        #reguloss += 1e1*regu2(parameters, 2,img)

        #loss = chi2loss + reguloss

        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward

        t.set_postfix({'loss': loss.item(), 'chi2loss': chi2loss.item(), 'reguloss': reguloss.item()})

    parameters, final_output = evaluate(parameters,xl,img, model)
    outparams = parameters.detach().numpy().reshape(img.shape[0],img.shape[1],11)
    return outparams, final_output



# ====================================================================
def Gaussian_model(x, params):
    x = torch.from_numpy(np.array(x.astype(np.float32)))
    sigma = torch.abs(params[:,1])
    return torch.pow(10,params[:,0])*torch.exp(-(x-params[:,2])**2./(2*(sigma+1e-6)**2.))

# ========================================================================================================

def cloud_model(x, params):
    x = torch.from_numpy(np.array(x.astype(np.float32)))
    return cmt.model_synth(x, params)

# ====================================================================
def evaluate(out,xl,img,model):
    final_output = torch.tensor(())
    for ii in range(xl.shape[0]):
        final_output = torch.cat((final_output, model(xl[ii], out)), 0)
    final_output = torch.reshape(final_output, (img.shape[0],img.shape[1],img.shape[2]))
    return out, final_output



# ====================================================================
def regu2(out, ii,img):
    # Squared diference
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(output_smooth-mout[:,:,:,ii])**2.0)



'''ll = np.linspace(6499.0, 6501.,201)
ll0 = np.asarray([6500.0])
x = [ll,ll0]

p = np.array([0.5, 0.5, 100.0, 0.0, 10.0, -1, 0.2, 5.0, 20.0, 5.0, -5])
p = np.concatenate((p[None,:], p[None,:]), axis=0)
print(p.shape)
pt = torch.from_numpy(p.astype(np.float32))
print(pt.shape)
test = cmt.model_synth(x, pt)
print(test.shape)
print(test)

plt.figure(figsize=[9,5])
plt.plot(ll, test[0,:])
plt.plot(ll, test[1,:])
plt.savefig("testerino.png",bbox_inches='tight')'''


# ================================================================================================================
# Readind the data:
tofit = fits.open(sys.argv[1])[0].data[:,:,:,100:500]
xl = fits.open(sys.argv[1])[1].data[100:500]
ll0 = np.asarray([6500.0])
xl = [xl,ll0]
xl = np.asarray(xl)
print (xl)

# Transform data into pytorch object:
tofit_pt = torch.from_numpy(np.array(tofit.astype(np.float32)))

# ================================================================================================================
# Defining the parameters of the model in the FOV:
pars = torch.ones(tofit.shape[0]*tofit.shape[1], 11) # (NX x NY) x NP cube
# Initialize the starting values:
pars[:,0] = 0.5 #S0
pars[:,1] = 0.5 #S1
pars[:,2] = 100 # eta
pars[:,3] = 2.0 # los velocity
pars[:,4] = 10.0 # doppler width 
pars[:,5] = -1.0 # log a
# Cloud: 
pars[:,6] = 0.2 # S
pars[:,7] = 5.0 # tau
pars[:,8] = -2.0 # los velocity
pars[:,9] = 5.0 # Doppler width
pars[:,10] = -5.0 # log a

pars.requires_grad = True

# ====================================================================
# Send all the info to the optimization module inside the utils.py file:
optimizer = torch.optim.Adam([pars], lr=1e-3)
niter = 1
mymodel = cloud_model # Model from the above
outplot, final_output = optimization(optimizer=optimizer,niterations=niter,
                                        parameters=pars,model=mymodel,xl=xl,img=tofit)

# ====================================================================
#name = 'fit_torch_regu_on'

'''
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(20/2,10/2))
im = ax1.imshow(10.**outplot[:,:,0].T,cmap='gray',interpolation='nearest',vmin=10,vmax=200) #
ax1.set_title('Amplitude [counts]')
cbar = add_colorbar(im, aspect=30)
plt.locator_params(axis='y', nbins=4)
ax1.minorticks_on()
im2 = ax2.imshow(1e3*np.abs(outplot[:,:,1].T),cmap='cubehelix',interpolation='nearest',vmin=1e3*0.01,vmax=1e3*0.032)#,vmin=0.015,vmax=+0.03)#
ax2.set_title('Width [mA]')
cbar = add_colorbar(im2, aspect=35)
im3 = ax3.imshow(1e3*outplot[:,:,2].T,cmap='bwr',interpolation='nearest',vmin=1e3*-0.03,vmax=1e3*+0.03)#,
ax3.set_title('Center [mA]')
add_colorbar(im3, aspect=30)
plt.savefig(name+'.png',bbox_inches='tight',dpi=600)'''
