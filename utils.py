import math
import torch
import torch.nn.functional as F
import numpy as np

import sde_lib


def get_optimizer(config,params):
    if config.optim.optimizer == "Adam":
        optimizer = torch.optim.Adam(params,lr=config.optim.lr,betas=(config.optim.beta1,0.999),eps = config.optim.eps,weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')
    return optimizer


def get_model_fn(model,train=False):
    def model_fn(*args):
        if not train:
            model.eval()
            return model(*args)
        else:
            model.train()
            return model(*args)
        
    return model_fn
  

def get_score_fn(sde,model,train=False,continuous=False):
    model_fn = get_model_fn(model,train=train)

    if isinstance(sde,sde_lib.VPSDE) or isinstance(sde,sde_lib.subVPSDE):
        def score_fn(x,t):
            if continuous or isinstance(sde,sde_lib.subVPSDE):
                labels = t * 999
                score = model_fn(x,labels)[:,:x.shape[1],:]
                std = sde.marginal_prob(torch.zeros_like(x),t)[1]
            else:
                labels = t * (sde.N - 1)
                score = model_fn(x,labels)[:,:x.shape[1],:]
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
            
            score = -score / std[:,None,None]
            return score

    elif isinstance(sde,sde_lib.VESDE):
        def score_fn(x,t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x),t)[1]
            else:
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
            score = model_fn(x,labels)
            return score
    else:
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported")
	
    return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def get_dataset(dir,config):
    numpy_data = np.load(dir,allow_pickle=True).item()
    train_data = numpy_data['data']
    dataloader = torch.utils.data.DataLoader(train_data,config.data.batch_size,shuffle=True)
    _size, feat , seq_len = train_data.shape
    assert seq_len == config.data.seq_len and config.data.num_features == feat
    return dataloader