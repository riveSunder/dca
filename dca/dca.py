import numpy as np
import time

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
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    grad_x = F.conv2d(state_grid, sobel_x, padding=1, groups=16) #torch.tensor(state), torch.tensor(sobel_x))
    grad_y = F.conv2d(state_grid, sobel_y, padding=1, groups=16) #torch.tensor(state), torch.tensor(sobel_y))

    perception = torch.cat([state_grid, grad_x, grad_y], axis=1)

    return perception

def update(perception, weights_0, weights_1, bias_0, bias_1):
    # use conv2d with 1x1 kernels and groups = input channels, then sum
    # to effectively get dense nn for each location

    groups_0 = 1 #perception.shape[1]
    x = F.conv2d(perception, weights_0, padding=0, groups=groups_0) #, bias=bias_0)
    x = torch.relu(x)

    groups_1 = 1# 16
    #  weights.shape[1] = x.shape[1] / groups
    x = F.conv2d(x, weights_1, padding=0, groups=groups_1) #, bias=bias_1)

    # squash result from 0 to 1
    
    #x = torch.sigmoid(x-50)
    
    #x = (x - torch.min(x)) / torch.max(x - torch.min(x))
    x = torch.tanh(x)

    return x

#def update(perception, model):
#
#    perception = perception.reshape(perception.shape[0], \
#            perception.shape[1]* perception.shape[2] * perception.shape[3])
#
#    x = model(perception)

def stochastic_update(state_grid, perception, weights_0, weights_1, bias_0, bias_1, rate=0.5):

    # call update function, but randomly zero out some cell states
    updated_grid = update(perception, weights_0, weights_1, bias_0, bias_1)

    # can I just use dropout here?
    mask = torch.rand_like(updated_grid) < rate
    mask = mask.double()
    state_grid = mask * updated_grid

    return state_grid


def alive_masking(state_grid, threshold = 0.1):

    # alpha value must be greater than 0.1 to count as alive
    alive_mask = state_grid[:,3,:,:] > threshold #F.max_pool2d(state_grid[:,:,], kernel_size=3) > 0.1
    alive_mask = alive_mask.double()
    state_grid *= alive_mask


    return state_grid



if __name__ == "__main__":

    display = False
    my_dtype = torch.float64

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    h_dim = 128
    x_dim = 48
    y_dim = 16

    weights_0 = ( 1e-1 * torch.randn(h_dim, x_dim, 1, 1, dtype=my_dtype, device=device))
    weights_1 = ( 2e-1 * torch.randn(y_dim, h_dim, 1, 1, dtype=my_dtype, device=device))
    weights_0.requires_grad = True
    weights_1.requires_grad = True
    bias_0 = torch.zeros(h_dim, dtype=my_dtype, device=device, requires_grad=True)
    bias_1 = torch.zeros(y_dim, dtype=my_dtype, device=device, requires_grad=True)
    
    if(1):
        filename = "./data/aghast00.png"
        target = skimage.io.imread(filename)
        target = torch.tensor(target /255).double().to(device)

        lr = 1e-4
        disp_every = 100 #20
        batch_size = 4
        num_epochs = 100000
        num_steps = 64
        my_rate = 0.8

        optimizer = torch.optim.Adam([weights_0, weights_1, bias_0, bias_1], lr=lr)
    

        try:
            t0 = time.time()
            for epoch in range(num_epochs):
                #weights_0.zero_grad()
                #weights_1.zero_grad()
                loss = 0.0
                for batch in range(batch_size):
                    state_grid = torch.zeros((1,16,64,64)).double()
                    state_grid[:,0:4,32,32] += 1.0
                    state_grid = state_grid.to(device)

                    for ii in range(num_steps + np.random.randint(int(num_steps/2))):
                        state_grid = alive_masking(state_grid, threshold=0.1)
                        perception = sobel_conv2d(state_grid) 
                        state_grid = stochastic_update(state_grid, perception, weights_0, weights_1,\
                                bias_0, bias_1, rate=my_rate)

                    state_grid = alive_masking(state_grid, threshold=0.1)
                    pred = state_grid[0, 0:4, :, :].permute(1,2,0)

                    loss += torch.mean((pred-target)**2) / batch_size

                loss.backward()

                optimizer.step()
                if epoch % disp_every == 0:
                    num_alive = torch.sum(state_grid[0,3,:,:] > 0.1)
                    num_cells = state_grid.shape[2]*state_grid.shape[3]
                    print("num live cells: {} of {}".format(num_alive, num_cells))
                    print("ca grid stats: mean {:.2e}, min {:.2e}, max {:.2e}".format(\
                            torch.mean(state_grid).cpu().detach().numpy(), torch.min(state_grid).cpu().detach().numpy(), torch.max(state_grid).cpu().detach().numpy())) 
                    elapsed = time.time() - t0
                    print("loss at epoch {}: {:.3e}, elapsed: {:.3f}, per epoch {:.2e}"\
                            .format(epoch, loss, elapsed, elapsed/(1+epoch)))

        except KeyboardInterrupt:
            pass

        state_grid = torch.zeros((1,16,64,64)).double()
        state_grid[:,0:4,32,32] += 1.0
        state_grid = state_grid.to(device)

        for ii in range(128):
            state_grid = alive_masking(state_grid)
            img = state_grid[0, 0:4, :, :].permute(1,2,0) 
            img = img.cpu().detach().numpy()

            skimage.io.imsave("./output/state{}.png".format(ii), img)

            perception = sobel_conv2d(state_grid) 
            state_grid = stochastic_update(state_grid, perception, weights_0, weights_1,\
                    bias_0, bias_1, rate=my_rate)
        print("here")
