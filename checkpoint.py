import os
import torch
import logging

def restore_checkpoint(ckpt_dir, state, device):
  if ckpt_dir == None or not os.path.exists(ckpt_dir):
    logging.info(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['encoder'].load_state_dict(loaded_state['encoder'])
    state['decoder'].load_state_dict(loaded_state['decoder'])
    state['denoiser'].load_state_dict(loaded_state['denoiser'])
    state['encoder_optim'].load_state_dict(loaded_state['encoder_optim'])
    state['decoder_optim'].load_state_dict(loaded_state['decoder_optim'])
    state['denoiser_optim'].load_state_dict(loaded_state['denoiser_optim'])
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'encoder_optim': state['encoder_optim'].state_dict(),
    'decoder_optim': state['decoder_optim'].state_dict(),
    'denoiser_optim': state['denoiser_optim'].state_dict(),
    'encoder': state['encoder'].state_dict(),
    'decoder': state['decoder'].state_dict(),
    'denoiser': state['denoiser'].state_dict(),
  }
  torch.save(saved_state, ckpt_dir)