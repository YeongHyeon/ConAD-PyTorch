import torch

def mean_square_error(x, x_hat):

    try: x, x_hat = x.cpu(), x_hat.cpu()
    except: pass

    # regarded as multiplied mse with specific bias
    data_dim = len(x.shape)
    if(data_dim == 5):
        return torch.sum((x - x_hat)**2, dim=(1, 2, 3, 4))
    elif(data_dim == 4):
        return torch.sum((x - x_hat)**2, dim=(1, 2, 3))
    elif(data_dim == 3):
        return torch.sum((x - x_hat)**2, dim=(1, 2))
    elif(data_dim == 2):
        return torch.sum((x - x_hat)**2, dim=(1))
    else:
        return torch.sum((x - x_hat)**2)

def find_best_x(x_mulin, x_mulout):

    x_mulin, x_mulout = \
        x_mulin.cpu(), x_mulout.cpu()

    mse = mean_square_error(x_mulin, x_mulout)
    best_idx = torch.argmin(mse).item()

    return best_idx

def lossfunc_d(d_real, d_fake, d_best, d_others, num_h):

    d_real, d_fake, d_best, d_others = \
        d_real.cpu(), d_fake.cpu(), d_best.cpu(), d_others.cpu()

    l_real = -torch.log(d_real + 1e-12)
    l_fake = (torch.log(d_fake + 1e-12) + torch.log(d_best + 1e-12) + torch.log(d_others + 1e-12)) / (num_h + 1)
    loss_d = torch.abs(torch.mean(l_real + l_fake))

    return loss_d

def lossfunc_g(x, x_best, z_mu, z_sigma, loss_d):

    x, x_best, z_mu, z_sigma = \
        x.cpu(), x_best.cpu(), z_mu.cpu(), z_sigma.cpu()

    restore_error = -torch.sum(x * torch.log(x_best + 1e-12) + (1 - x) * torch.log(1 - x_best + 1e-12), dim=(1, 2, 3))
    kl_divergence = 0.5 * torch.sum(z_mu**2 + z_sigma**2 - torch.log(z_sigma**2 + 1e-12) - 1, dim=(1))

    ELBO = torch.mean(restore_error + kl_divergence)
    loss_g = ELBO - loss_d

    return loss_g
