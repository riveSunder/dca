import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import skimage
import skimage.io 
import matplotlib.pyplot as plt

def sobel_conv2d(state_grid):
    sobel_y = torch.tensor(np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]),\
            dtype=torch.float64)
    sobel_x = torch.tensor(np.array([[[[-1, 2, -1], [0, 0, 0], [1, 2, 1]]]]), \
            dtype=torch.float64)

    sobel_x = sobel_x * torch.ones((state_grid.shape[1], 1, 3,3))
    sobel_y = sobel_y * torch.ones((state_grid.shape[1], 1, 3,3))
    ## use Fourier transform 
    #grad_x = np.fft.ifft2(np.fft.fft2(state)*np.fft.fft2(sobel_x, state.shape[0], state.shape[1]))
    #grad_y = np.fft.ifft2(np.fft.fft2(state)*np.fft.fft2(sobel_y, state.shape[0], state.shape[1]))
    
    # apply Sobel filters
    if len(state_grid.shape) == 2:
        state = state.reshape(1,1,state_grid.shape[0], state_grid.shape[1])

    grad_x = F.conv2d(state_grid, sobel_x, padding=1, groups=16) #torch.tensor(state), torch.tensor(sobel_x))
    grad_y = F.conv2d(state_grid, sobel_y, padding=1, groups=16) #torch.tensor(state), torch.tensor(sobel_y))

    perception = torch.cat([state_grid, grad_x, grad_y], axis=1)

    return perception

def update(perception, weights_0, weights_1):
    # use conv2d with 1x1 kernels and groups = input channels, then sum
    # to effectively get dense nn for each location

    groups_0 = perception.shape[1]
    x = F.conv2d(perception, weights_0, padding=0, groups=groups_0)
    x = F.relu(x)

    groups_1 = 1 #x.shape[1]
    #  weights.shape[1] = x.shape[1] / groups
    x = F.conv2d(x, weights_1, padding=0, groups=groups_1)

    # squash result from 0 to 1
    
    
    return x

#def update(perception, model):
#
#    perception = perception.reshape(perception.shape[0], \
#            perception.shape[1]* perception.shape[2] * perception.shape[3])
#
#    x = model(perception)

def stochastic_update(state_grid, perception, weights_0, weights_1, rate=0.5):

    # call update function, but randomly zero out some cell states
    updated_grid = update(perception, weights_0, weights_1)

    # can I just use dropout here?
    mask = torch.rand_like(updated_grid) < rate
    mask = mask.double()
    state_grid = mask * updated_grid

    return state_grid


def alive_masking(state_grid, threshold = 0.5):

    # alpha value must be greater than 0.1 to count as alive
    alive_mask = state_grid[:,3,:,:] > threshold #F.max_pool2d(state_grid[:,:,], kernel_size=3) > 0.1
    alive_mask = alive_mask.double()
    state_grid *= alive_mask


    return state_grid



if __name__ == "__main__":

    display = False
    my_dtype = torch.float64

    state_grid = torch.zeros((1,16,64,64))
    state_grid[:,:,32,32] = 0.5
    state_grid = state_grid.double()
    weights_0 = (3*torch.rand(48, 1, 1, 1, dtype=my_dtype)).double().requires_grad_()
    weights_1 = (3*torch.rand(16, 48, 1, 1, dtype=my_dtype)).double().requires_grad_()

    result = sobel_conv2d(state_grid)

    if (display):
        plt.figure()
        plt.subplot(221)
        plt.imshow(state_grid[0,0,:,:])
        plt.title("state")
        plt.subplot(222)
        plt.imshow(result[0,0,:,:])
        plt.subplot(223)
        plt.imshow(result[0,1,:,:])
        plt.subplot(224)
        plt.imshow(result[0,2,:,:])
        plt.show()
    
    state_grid = alive_masking(state_grid)

    #updated_perception = update(result, weights_0, weights_1)
    state_grid = stochastic_update(state_grid, result, weights_0, weights_1)

    if(1):
        filename = "./data/aghastgb.tif"
        target = skimage.io.imread(filename)
        target = torch.tensor(target /255).double()
        lr = 1e-3
        disp_every = 100
        num_epochs = 1000

        optimizer = torch.optim.SGD([weights_0, weights_1], lr=lr)
    

        for epoch in range(num_epochs):
            #weights_0.zero_grad()
            #weights_1.zero_grad()
            state_grid = torch.zeros((1,16,64,64))
            state_grid[:,:,30:34,30:34] += 1.0
            for ii in range(128):
                state_grid = alive_masking(state_grid)
                state_grid = stochastic_update(state_grid, result, weights_0, weights_1)

            pred = state_grid[0, 0:3, :, :].permute(1,2,0)

            loss = torch.mean((pred-target)**2)
            loss.backward()
            optimizer.step()
            if epoch % disp_every == 0:
                print("loss at epoch {}: {:.3e}".format(epoch, loss))

        
        state_grid = torch.zeros((1,16,64,64))
        state_grid[:,:,30:34,30:34] += 1.0

        import pdb; pdb.set_trace()

        for ii in range(128):
            state_grid = stochastic_update(state_grid, result, weights_0, weights_1)
            state_grid = alive_masking(state_grid)
            img = state_grid[0, 0:3, :, :].permute(1,2,0) 
            img = img.detach().numpy()

            plt.figure()
            plt.imshow(img)
            plt.savefig("./output/state{}.png".format(ii))
            plt.clf()
            plt.cla()
        print("here")
