import numpy as np
import time
import os

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import skimage
import skimage.io 
import skimage.transform 
import matplotlib.pyplot as plt

def get_perception(state_grid, device="cpu"):




    my_dim = state_grid.shape[1]
    moore = torch.tensor(np.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]]),\
            dtype=torch.float64)
    sobel_y = torch.tensor(np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]),\
            dtype=torch.float64)
    sobel_x = torch.tensor(np.array([[[[-1, 2, -1], [0, 0, 0], [1, 2, 1]]]]), \
            dtype=torch.float64)

    moore /= torch.sum(moore)

    sobel_x = sobel_x * torch.ones((state_grid.shape[1], 1, 3,3))
    sobel_y = sobel_y * torch.ones((state_grid.shape[1], 1, 3,3))
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    moore = moore * torch.ones((state_grid.shape[1], 1, 3,3))
    moore = moore.to(device)

    grad_x = F.conv2d(state_grid, sobel_x, padding=1, groups=my_dim)
    grad_y = F.conv2d(state_grid, sobel_y, padding=1, groups=my_dim)

    moore_neighborhood = F.conv2d(state_grid, moore, padding=1, groups=my_dim)

    perception = torch.cat([state_grid, moore_neighborhood, grad_x + grad_y], axis=1)

    return perception

def update(perception, weights_0, weights_1, bias_0, bias_1):

    groups_0 = 1
    use_bias = 1
    if use_bias:
        x = F.conv2d(perception, weights_0, padding=0, groups=groups_0, bias=bias_0)
    else:
        x = F.conv2d(perception, weights_0, padding=0, groups=groups_0)

    x = torch.atan(x)

    groups_1 =  1

    if use_bias:
        x = F.conv2d(x, weights_1, padding=0, groups=groups_1, bias=bias_1)
    else:
        x = F.conv2d(x, weights_1, padding=0, groups=groups_1)

    x = torch.sigmoid(x)

    return x

def stochastic_update(state_grid, perception, weights_0, weights_1, bias_0, bias_1, rate=0.5):

    # call update function, but randomly zero out some cell states
    updated_grid = update(perception, weights_0, weights_1, bias_0, bias_1)

    # can I just use dropout here?
    mask = torch.rand_like(updated_grid) < rate
    mask = mask.double()
    state_grid = mask * updated_grid + (1 - mask) * state_grid

    return state_grid


def alive_masking(state_grid, threshold = 0.1):

    # in case there is no alpha channel
    # this should only take place when loading images from disk
    if state_grid.shape[1] == 3 and state_grid.shape[0] == 1:
        temp = torch.ones_like(state_grid[0,0,:,:])

        temp[torch.mean(state_grid[0], dim=0) > 0.99] *= 0.0
        state_grid = torch.cat([state_grid, temp.unsqueeze(0).unsqueeze(0)], dim=1)

    # alpha value must be greater than 0.1 to count as alive
    alive_mask = state_grid[:,3:4,:,:] > threshold
    alive_mask = alive_mask.double()
    state_grid *= alive_mask


    return state_grid

def take_a_bite(img_tensor, my_radius = 6.0):

    num_imgs, dim_ch, dim_x, dim_y = img_tensor.shape

    my_offset = 10

    coords = np.random.randint(low=(my_offset,my_offset),
            high=(dim_x-my_offset, dim_y-my_offset), size=(num_imgs,2))

    x = np.arange(0, dim_x) #linspace(-dim_x/2, dim_x/2-1, dim_x)
    y = np.arange(0, dim_y) #np.linspace(-dim_x/2, dim_x/2-1, dim_y)

    xx, yy = np.meshgrid(x,y)

    for ii in range(num_imgs):

        rr = np.sqrt((xx - coords[ii,0])**2 + (yy - coords[ii,1])**2)
        ablation_mask = np.ones((dim_x, dim_y))
        ablation_mask[rr <= my_radius] = 0.0


        img_tensor[ii,:,:,:] *= ablation_mask[np.newaxis,:,:]

    
    return img_tensor

def evaluate_loss(weights_0, weights_1, bias_0, bias_1, target_batch, \
        y_dim=16, min_r=4.0, mask_noise=0.5, num_steps=12, device="cpu"):

    target_batch_size = target_batch.shape[0]
    dim_x, dim_y = target_batch.shape[2], target_batch.shape[3]
    xx, yy = np.meshgrid(np.linspace(-dim_x // 2, dim_x // 2, dim_x), \
            np.linspace(-dim_x // 2 , dim_y // 2, dim_y))
    rr = np.sqrt(xx**2 + yy**2)
    rr = rr[np.newaxis, :, :]
    rr = rr * np.ones((y_dim, dim_x, dim_y))
    my_rate = 0.9
    
    weights_0 = weights_0.to(device)
    weights_1 = weights_1.to(device)
    bias_0 = bias_0.to(device)
    bias_1 = bias_1.to(device)
    
    target_batch = target_batch.to(device)

    with torch.no_grad():
        
        state_grid = torch.zeros((target_batch_size, y_dim, dim_x, dim_y)).double()
        state_grid[:,0:4] = target_batch \
                + torch.rand_like(target_batch) * mask_noise\
                * (torch.rand_like(target_batch) > mask_noise)

        r_mask = np.zeros((y_dim, dim_x, dim_y))
        r_mask[rr <= min_r] = 1.0
        state_grid[0] *= r_mask

        state_grid = state_grid.to(device)
        
        #inp = 1.0 * state_grid[0, 0:4, :, :].permute(1,2,0).cpu().numpy()
        #tgt = target_batch[0].permute(1,2,0).cpu().numpy()

        for step in range(num_steps):

            perception = get_perception(state_grid, device=device) 
            state_grid = stochastic_update(state_grid, perception, weights_0, weights_1,\
                    bias_0, bias_1, rate=my_rate)

        pred = state_grid[:,0:4,:,:]
        loss = torch.mean(torch.abs(pred-target_batch)\
                + torch.abs((pred-target_batch)**2)) 

    return loss

def save_things(weights_0, weights_1, bias_0, bias_1, epoch=0, target=None, y_dim=8):

    
    with torch.no_grad():
        
        state_grid = torch.zeros((1, y_dim, dim_x, dim_y)).double()
        state_grid[0,0:4] = target \
                + torch.rand_like(target) * grid_mask\
                * (torch.rand_like(target) > grid_mask)\

        r_mask = np.zeros((y_dim, dim_x, dim_y))
        r_mask[rr <= (radius + np.random.random())] = 1.0
        state_grid[0] *= r_mask
        state_grid = take_a_bite(state_grid, bite_radius)

        #state_grid[:,0:4,32,32] += 1.0
        inp = 1.0 * state_grid[0, 0:4, :, :].permute(1,2,0)
        state_grid = state_grid.to(device)

        np.save("./models/{}_dca_model.npy".format(args.exp_name),\
                [weights_0, weights_1, bias_0, bias_1])
        
        tgt = target.permute(1,2,0).cpu().numpy()
        for ii in range(num_steps*2+1):
            state_grid = alive_masking(state_grid)
            img = state_grid[0, 0:3, :, :].permute(1,2,0) 
            img2 = state_grid[0, 0:4, :, :].permute(1,2,0) 
            img = img.cpu().detach().numpy()
            img2 = img2.cpu().detach().numpy()
            img = np.array(img)
            img2 = np.array(img2)

            if target is not None and ii == (num_steps):
                fig = plt.figure(figsize=(9,9))
                plt.subplot(221)
                plt.imshow(inp)
                plt.title("input")
                plt.subplot(222)
                plt.imshow(tgt)
                plt.title("target")
                plt.subplot(223)
                plt.imshow(img2)
                plt.title("output epoch {}, step {}".format(epoch, ii))
                plt.subplot(224)

                plt.imshow(np.mean(np.abs(img2-tgt), axis=2))
                plt.title("absolute mean difference")
                plt.colorbar()
                plt.savefig("output/{}_epoch{}_comparison.png".format(args.exp_name, epoch))
                plt.close(fig)

            #skimage.io.imsave("./output/epoch{}state{}.png".format(epoch, ii), img)
            fig = plt.figure(figsize=(9,9))

            plt.imshow(state_grid[0,0:4,:,:].permute(1,2,0).detach().cpu())
            plt.title("step {}".format(ii))
            plt.savefig("./output/{}_epoch{}_state{}.png".format(args.exp_name, epoch, ii))
            plt.close(fig)
            
            perception = get_perception(state_grid, device=device) 
            state_grid = stochastic_update(state_grid, perception, weights_0, weights_1,\
                    bias_0, bias_1, rate=my_rate)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-x", "--exp_name", type=str,\
            default="default_exp", help="name of experiment")
    parser.add_argument("-d", "--data_path", type=str,\
            default="one_pokemon/", help="data folder")

    parser.add_argument("-l", "--load", type=bool,\
            default=False, help="load previous model")
    parser.add_argument("-p", "--persistence", type=bool,\
            default=False, help="use persistence loss")

    parser.add_argument("-r", "--radius", type=float,\
            default=6.0, help="minimum stopdown radius")
    parser.add_argument("-b", "--bite", type=float,\
            default=4.0, help="maximum bite radius")
    parser.add_argument("-n", "--noise", type=float,\
            default=0.20, help="maximum additive noise")


    args = parser.parse_args()

    display = False
    my_dtype = torch.float64

    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"

    h_dim = 64
    y_dim = 16
    x_dim = y_dim * 3

    if(args.load):
        params = np.load("{}_dca_model.npy".format(args.exp_name), allow_pickle=True)
        weights_0 = params[0]
        weights_1 = params[1]
        bias_0 = params[2]
        bias_1 = params[3]
    elif(1):
        temp = 100
        weights_0 = ( temp * torch.randn(h_dim, x_dim, 1, 1, dtype=my_dtype, device=device)\
                / (h_dim*x_dim))
        weights_1 = ( temp * torch.randn(y_dim, h_dim, 1, 1, dtype=my_dtype, device=device) \
                / (y_dim*h_dim))
        weights_0.requires_grad = True
        weights_1.requires_grad = True
        bias_0 = torch.zeros(h_dim, dtype=my_dtype, device=device, requires_grad=True)
        bias_1 = torch.zeros(y_dim, dtype=my_dtype, device=device, requires_grad=True)

    
    if(1):

        training_dir = "./data/{}".format(args.data_path)
        dir_list = os.listdir(training_dir)
        targets = torch.Tensor().double().to(device)
        
        dim_x, dim_y = 64, 64

        for filename in dir_list:

            target = skimage.io.imread(training_dir + filename)

            target = skimage.transform.resize(target, (dim_x, dim_y))

            target = torch.tensor(target).double().to(device)
            target = target.permute(2,0,1).unsqueeze(0)

            target = alive_masking(target)

            targets = torch.cat([targets, target], dim=0)
            
        num_samples = targets.shape[0]

        lr = 3e-4
        disp_every = 1000 #20
        batch_size = 1
        num_epochs = 60000
        num_steps = 2
        max_steps = 16
        my_rate = 0.8
        l2_reg = 1e-6

        xx, yy = np.meshgrid(np.linspace(-dim_x // 2, dim_x // 2, dim_x), \
                np.linspace(-dim_x // 2 , dim_y // 2, dim_y))

        rr = np.sqrt(xx**2 + yy**2)
        rr = rr[np.newaxis, :, :]
        rr = rr * np.ones((y_dim, dim_x, dim_y))
        
        start_r = dim_x / 1
        radius = start_r * 1.0
        r_decay = 0.99
        grid_mask = 0.01
        mask_decay = 0.0005
        bite_increase = 0.1
        bite_radius = 0.00

        bite_max = args.bite
        max_mask = args.noise
        min_r = args.radius

        train_persistence = args.persistence

        optimizer = torch.optim.Adam([weights_0, weights_1, bias_0, bias_1], lr=lr)
        
        try:
            t0 = time.time()
            exp_id = str(t0)[-16:-7]
            progress = {}
            progress["loss"] = [] 
            progress["epoch"] = [] 
            progress["time"] = [] 
            for epoch in range(num_epochs):
                
                if weights_0.grad is not None:
                    weights_0.requires_grad = False
                    weights_0.grad *= 0.0
                    weights_0.requires_grad = True
                    weights_1.requires_grad = False
                    weights_1.grad *= 0.0
                    weights_1.requires_grad = True

                if bias_0.grad is not None:
                    bias_0.requires_grad = False
                    bias_0.grad *= 0.0
                    bias_0.requires_grad = True
                    bias_1.requires_grad = False
                    bias_1.grad *= 0.0
                    bias_1.requires_grad = True


                for batch in range(num_samples // batch_size): #batch_size):

                    state_grid = torch.zeros((batch_size, y_dim, dim_x, dim_y)).double()
                    
                    indices = np.random.choice(np.arange(num_samples), \
                            p=[1/num_samples] * num_samples,\
                            size = batch_size)

                    target_batch = torch.zeros(batch_size, 4, dim_x,dim_y)
                    for jj in range(batch_size):
                        # omit pixels 
                        target_batch[jj,:,:,:] = targets[indices[jj]]
                        state_grid[jj,0:4] = targets[indices[jj]] \
                                + torch.rand_like(target) * grid_mask \
                                * (torch.rand_like(target) > grid_mask)\
                        
                        r_mask = np.zeros((y_dim, dim_x, dim_y))
                        r_mask[rr <= (radius + np.random.random())] = 1.0
                        state_grid[jj] *= r_mask

                    state_grid = take_a_bite(state_grid, bite_radius)
                    state_grid = state_grid.to(device)
                    target_batch = target_batch.to(device)

                    extra_steps = 8 #max([1, int(num_steps*0.5)])

                    loss = 0.0
                    for ii in range(num_steps + extra_steps): # + np.random.randint(int(num_steps/2))):
                        state_grid = alive_masking(state_grid)

                        if ii >= (num_steps):

                            pred = state_grid[:,0:4,:,:]
                            loss += torch.mean(torch.abs(pred-target_batch)\
                                    + torch.abs(pred-target_batch)**2) / extra_steps

                        perception = get_perception(state_grid, device=device) 
                        state_grid = stochastic_update(state_grid, perception, weights_0, weights_1,\
                                bias_0, bias_1, rate=my_rate)

                    if train_persistence:
                        # include a separate 
                        for ii in range(num_steps + extra_steps): # + np.random.randint(int(num_steps/2))):
                            state_grid = alive_masking(state_grid)

                            if ii >= (num_steps):

                                pred = state_grid[:,0:4,:,:]
                                loss += torch.mean(torch.abs(pred-target_batch)\
                                        + torch.abs(pred-target_batch)**2) / extra_steps

                            perception = get_perception(state_grid, device=device) 
                            state_grid = stochastic_update(state_grid, perception, weights_0, weights_1,\
                                    bias_0, bias_1, rate=my_rate)


                    for params in [weights_0, weights_1, bias_0, bias_1]:
                        loss += l2_reg * torch.sum(torch.abs(params)**2)

                    loss.backward()

                    optimizer.step()

                    if loss <= 0.025: #epoch % 20 == 0:
                        grid_mask = min([max_mask, grid_mask + mask_decay])
                        radius = max([min_r, radius * r_decay])

                        num_steps = min([max_steps, num_steps + 1 ])

                        bite_radius = min([bite_max, bite_radius + bite_increase])



                if epoch % disp_every == 0:

                    print("grid_mask={:.2f}, num_steps={},radius={:.2f}, bite={:.2f}".format(\
                            grid_mask, num_steps, radius, bite_radius))

                    
                    num_alive = torch.sum(state_grid[0,3,:,:] > 0.1)
                    num_cells = state_grid.shape[2]*state_grid.shape[3]
                    print("num live cells: {} of {}".format(num_alive, num_cells))
                    print("ca grid stats: mean {:.2e}, min {:.2e}, max {:.2e}".format(\
                            torch.mean(state_grid).cpu().detach().numpy(), torch.min(state_grid).cpu().detach().numpy(), torch.max(state_grid).cpu().detach().numpy())) 
                    elapsed = time.time() - t0
                    print("loss at epoch {}: {:.3e}, elapsed: {:.3f}, per epoch {:.2e}"\
                            .format(epoch, loss, elapsed, elapsed/(1+epoch)))

                    save_things(weights_0, weights_1, bias_0, bias_1, epoch=epoch,\
                            target=targets[indices[0]], y_dim=y_dim)

                    loss = evaluate_loss(weights_0, weights_1, bias_0, bias_1,\
                            targets)

                    wall_time = time.time() - t0
                    progress["loss"].append(loss)
                    progress["epoch"].append(epoch)
                    progress["time"].append(wall_time)

                    np.save("results/{}_progress_{}.npy".format(args.exp_name, exp_id),\
                            progress, allow_pickle=True)

        except KeyboardInterrupt:
            pass

        save_things(weights_0, weights_1, bias_0, bias_1, epoch=epoch, \
                target=targets[indices[0]], y_dim=y_dim)

        loss = evaluate_loss(weights_0, weights_1, bias_0, bias_1,\
                targets)

        wall_time = time.time() - t0
        progress["loss"].append(loss)
        progress["epoch"].append(epoch)
        progress["time"].append(wall_time)
        np.save("results/{}_progress_{}.npy".format(args.exp_name, exp_id), progress, allow_pickle=True)

        print("here")
