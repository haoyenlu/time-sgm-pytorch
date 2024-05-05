import torch
import numpy as np


import utils as mutils
import models
import sde_lib
import configs




def get_sde_loss_fn(sde,train,reduce_mean=True,continuous=True,likelihood_weighting=True,eps=1e-5):
    reduce_op = torch.mean if reduce_mean else lambda *args,**kwargs: 0.5 * torch.sum(*args,**kwargs)

    def loss_fn(model,batch):
        score_fn = mutils.get_score_fn(sde,model,train=train,continuous=continuous)
        t = torch.rand(batch.shape[0],device=batch.device) * (sde.T - eps) + eps # Uniform distributed timestep
        z = torch.randn_like(batch) # noise
        mean, std = sde.marginal_prob(batch,t)
        perturbed_data = mean + std[:,None,None] * z 
        # concatenate with previous time step
        # prev = torch.zeros_like(batch)
        # prev[:,1:,:] = batch[:,0:-1,:]
        # perturbed_data = torch.cat([perturbed_data,prev],dim=1) 
        score = score_fn(perturbed_data,t) # using perturbed data with time embedding to calculate score value
        if not likelihood_weighting:
            losses = torch.square(score * std[:,None,None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0],-1),dim=-1)
        else:
            g2 = sde.sde(torch.zeros_kile(batch),t)[1] ** 2
            losses = torch.square(score + z / std[:,None,None])
            losses = reduce_op(losses.reshape(losses.shape[0],-1),dim=-1) * g2
        
        loss = torch.mean(losses)
        return loss

    return loss_fn

def get_recon_loss_fn(train,reduce_mean = True):
    reduce_op = torch.mean if reduce_mean else lambda *args,**kwargs: 0.5 * torch.sum(*args,**kwargs)

    def loss_fn(encoder,decoder,batch):
            encoder_fn = mutils.get_model_fn(encoder,train=train)
            decoder_fn = mutils.get_model_fn(decoder,train=train)

            output = encoder_fn(batch)
            z = torch.randn((output.shape),device=batch.device)
            recon_output  = decoder_fn(z)

            losses = torch.square(batch - recon_output)
            losses = reduce_op(losses.reshape(losses.shape[0],-1),dim=-1)
            loss = torch.mean(losses)
            return loss
    return loss_fn


def get_ed_step_fn(train,reduce_mean=True):
    loss_fn = get_recon_loss_fn(train,reduce_mean)
    def step_fn(state,batch):
        encoder = state['encoder']
        decoder = state['decoder']
        encoder_optim = state['encoder_optim']
        decoder_optim = state['decoder_optim']
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        loss = loss_fn(encoder,decoder,batch)

        if train:
            loss.backward()
            encoder_optim.step()
            decoder_optim.step()

        return loss
        
    return step_fn
        


def get_sde_step_fn(sde,train,optimize_fn=None,reduce_mean=True,continuous=True,likelihood_weighting=False,use_alt=False):
    loss_fn = get_sde_loss_fn(sde,train,reduce_mean=reduce_mean,continuous=continuous,likelihood_weighting=likelihood_weighting)

    def step_fn(state,batch):
        encoder = state['encoder']
        encoder_fn = mutils.get_model_fn(encoder,train=train)
        batch = encoder_fn(batch)
        denoiser = state['denoiser']
        denoiser_optim = state['denoiser_optim']

        denoiser_optim.zero_grad()
        loss = loss_fn(denoiser,batch.permute(0,2,1))
        if train:
            loss.backward()
            denoiser_optim.step()

        return loss

    return step_fn

