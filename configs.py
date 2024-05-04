import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 8
  training.n_iters = 1300001
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = True
  training.sde = 'subvpsde'



  # data
  config.data = data = ml_collections.ConfigDict()
  data.seq_len = 320
  data.num_features = 9

  # model
  config.model = model = ml_collections.ConfigDict()
  config.model.encoder = ml_collections.ConfigDict()
  model.sigma_max = 90
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20
  

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8


  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config