import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

def sobel_conv2d(state):
    sobel_y = torch.tensor(np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]),\
            dtype=torch.float64)
    sobel_x = torch.tensor(np.array([[[[-1, 2, -1], [0, 0, 0], [1, 2, 1]]]]), \
            dtype=torch.float64)

    ## use Fourier transform 
    #grad_x = np.fft.ifft2(np.fft.fft2(state)*np.fft.fft2(sobel_x, state.shape[0], state.shape[1]))
    #grad_y = np.fft.ifft2(np.fft.fft2(state)*np.fft.fft2(sobel_y, state.shape[0], state.shape[1]))
    
    # apply Sobel filters
    if len(state.shape) == 2:
        state = state.reshape(1,1,state.shape[0], state.shape[1])

    grad_x = F.conv2d(state, sobel_x, padding=1) #torch.tensor(state), torch.tensor(sobel_x))
    grad_y = F.conv2d(state, sobel_y, padding=1) #torch.tensor(state), torch.tensor(sobel_y))

    #

    perception = torch.cat([state, grad_x, grad_y], axis=1)


    return perception


if __name__ == "__main__":

    state = torch.tensor(np.random.random((1,1,64,64)))


    result = sobel_conv2d(state)

    plt.figure()
    plt.subplot(221)
    plt.imshow(state[0,0,:,:])
    plt.title("state")
    plt.subplot(222)
    plt.imshow(result[0,0,:,:])
    plt.subplot(223)
    plt.imshow(result[0,1,:,:])
    plt.subplot(224)
    plt.imshow(result[0,2,:,:])
    plt.show()

