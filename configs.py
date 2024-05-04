import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 8
  training.ed_iters = 50000
  training.main_iters = 40000
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = True
  training.sde = 'subvpsde'
  training.log_freq = 100
  training.checkpoint_freq = 1000


  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.num_sample = 200



  # data
  config.data = data = ml_collections.ConfigDict()
  data.batch_size = 8
  data.seq_len = 320
  data.num_features = 9

  # model
  config.model = model = ml_collections.ConfigDict()
  # Encoder
  model.encoder_hidden_dim = 1
  model.encoder_num_layer = 4
  # Decoder
  model.decoder_hidden_dim = 1
  model.decoder_num_layer = 4
  # Denoiser
  model.denoiser_hidden_dim = [16,32,64,128]
  # Score
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