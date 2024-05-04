import math
import torch
import torch.nn.functional as F

import sde_lib

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