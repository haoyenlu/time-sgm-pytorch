
import torch
from torch.utils.tensorboard import SummaryWriter
import configs
from models import Encoder, Decoder, Denoiser
import losses
import utils
import sde_lib
import sampling
import checkpoint
import argparse
import logging
import numpy as np
import os 
        
def train(args,config):
    # Create directories
    os.makedirs(args.ckptDir,exist_ok=True)
    os.makedirs(args.tfDir,exist_ok=True)
    os.makedirs(args.sampleDir,exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=args.tfDir)


    # initialize model
    encoder = Encoder(config).to(config.device)
    decoder = Decoder(config).to(config.device)
    denoiser = Denoiser(config).to(config.device)

    encoder_optim = utils.get_optimizer(config,encoder.parameters())
    decoder_optim = utils.get_optimizer(config,decoder.parameters())
    denoiser_optim= utils.get_optimizer(config,denoiser.parameters())

    state = dict(encoder=encoder,decoder=decoder,denoiser=denoiser,encoder_optim=encoder_optim,decoder_optim=decoder_optim,denoiser_optim=denoiser_optim)
    
    # load checkpoint if any
    checkpoint.restore_checkpoint(args.ckpt,state,config.device)

    
    # get SDE
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min,beta_max=config.model.beta_max,N=config.model.num_scales)
    sampling_eps = 1e-3


    # get Step function
    ed_step = losses.get_ed_step_fn(train=True,reduce_mean=True)
    sde_step = losses.get_sde_step_fn(sde,train=True)

    # get Sampling function
    sampling_shape = (config.sampling.num_sample,config.model.encoder_hidden_dim,config.data.seq_len)
    sampling_fn = sampling.get_sampling_fn(config,sde,sampling_shape,sampling_eps)

    # get data
    dataloader = utils.get_dataset(args.dir,config)
    train_iter = iter(dataloader)


    logging.info("Start Training encoder-decoder")
    # encoder-decoder training loop
    for step in range(config.training.ed_iters):
        batch = torch.from_numpy(next(train_iter))._numpy().to(config.device).float().permute(0,2,1)
        recon_loss = ed_step(state,batch)

        if step % config.training.log_freq == 0:
            logging.info(f"Step: {step}, Loss: {recon_loss.item()}")
            writer.add_scalar("Reconstruction Training Loss",recon_loss.item(),step)
        
        if step != 0 and step % config.training.checkpoint_freq == 0:
            checkpoint.save_checkpoint(os.path.join(args.ckptDir,f'checkpoint_{step}'),state)


    logging.info("Start Training score denoiser")
    # score diffusion training loop              
    for step in range(config.training.main_iters):
        batch = torch.from_numpy(next(train_iter))._numpy().to(config.device).float().permute(0,2,1)
        sde_loss = sde_step(state,batch)

        if step % config.training.log_freq == 0:
            logging.info(f"Step: {step}, Loss: {sde_loss.item()}")
            writer.add_scaler("Diffusion Training Loss",sde_loss.item(),step)

        if step != 0 and step % config.training.checkpoint_freq == 0:
            checkpoint.save_checkpoint(os.path.join(args.ckptDir,f'checkpoint_{step}'),state)

    
    # get samples
    samples, n = sampling_fn(denoiser)
    samples = decoder(samples.permute(0,2,1))
    samples = samples.detach().cpu().numpy()
    np.save(f"{args.sampleDir}/samples.npy",samples)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,help="Patient Dataset path",default=None)
    parser.add_argument('--ckptDir',type=str,help="Save Checkpoint Directory",default='./ckpt')
    parser.add_argument('--sampleDir',type=str,default='./samples')
    parser.add_argument('--tfDir',type=str,default='./runs')
    parser.add_argument('--ckpt',type=str,default=None)

    args = parser.parse_args()

    config = configs.get_default_configs()
    train(args,config)