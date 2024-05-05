import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import utils as mutils

class Encoder(nn.Module):
    def __init__(self,config): # input_size : num_feature
        super(Encoder,self).__init__()
        self.hidden_dim = config.model.encoder_hidden_dim
        self.num_layer = config.model.encoder_num_layer
        self.lstm = nn.LSTM(
            config.data.num_features,
            config.model.encoder_hidden_dim,
            config.model.encoder_num_layer,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self,x):
        outputs, (hidden,cell) = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder,self).__init__()
        self.lstm = nn.LSTM(
            config.model.encoder_hidden_dim,
            config.model.decoder_hidden_dim,
            config.model.decoder_num_layer,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(config.model.decoder_hidden_dim,config.data.num_features)
    
    def forward(self,x):
        output, (hidden,cell) = self.lstm(x)
        output = self.fc(output)
        return output
    

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel,stride,padding="same"):
        super(ConvBlock,self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_ch,out_ch,kernel_size=kernel,stride=stride,padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self,X):
        return self.model(X)





class Denoiser(nn.Module):
    def __init__(self,config):
        super(Denoiser,self).__init__()
        
        assert config.data.seq_len % 2 ** len(config.model.denoiser_hidden_dim) == 0

        self.hidden_dim = config.model.denoiser_hidden_dim
        self.input_dim = config.model.encoder_hidden_dim

        all_modules = []
        self.down = nn.AvgPool1d(kernel_size=5,padding=5//2,stride=2)
        self.up = nn.Upsample(scale_factor=2)

        # Downsampling block
        prev_ch = self.input_dim
        for dim in self.hidden_dim[:-1]:
            all_modules.append(self._make_conv_block(prev_ch,dim,kernel=5,stride=1,padding="same"))
            prev_ch = dim
        
        # Middle block
        all_modules.append(self._make_conv_block(prev_ch,self.hidden_dim[-1],kernel=5,stride=1,padding="same"))
        
        # Upsampling block
        prev_ch = self.hidden_dim[-1]
        for dim in reversed(self.hidden_dim[:-1]):
            all_modules.append(ConvBlock(prev_ch,dim,kernel=5,stride=1,padding="same"))
            all_modules.append(self._make_conv_block(2*dim,dim,kernel=5,stride=1,padding="same"))
            prev_ch = dim
        
        all_modules.append(nn.Conv1d(prev_ch,self.input_dim,kernel_size=5,stride=1,padding="same"))

        self.act = nn.ReLU() # Test with Relu and Tanh
        self.all_modules = nn.ModuleList(all_modules)

        self.temp_fc = nn.Sequential(
            nn.Linear(self.input_dim,self.input_dim),
            nn.ReLU()
        )

    def _make_conv_block(self,in_ch,out_ch,kernel,stride,padding="same"):
        model = []
        model.append(ConvBlock(in_ch,out_ch,kernel=kernel,stride=stride,padding=padding))
        model.append(ConvBlock(out_ch,out_ch,kernel,stride,padding))
        return nn.Sequential(*model)

    def forward(self,X,labels=None):
        xs = []
        _x = X

        if labels is not None:
            timesteps = labels
            temb = self.get_timestep_embedding(timesteps,self.input_dim)
            _x =_x + self.temp_fc(temb)[:,:,None]


        m_idx = 0
        for i in range(len(self.hidden_dim) - 1):
            _x = self.all_modules[m_idx](_x)
            m_idx += 1
            xs.append(_x)
            _x = self.down(_x)
        
        _x = self.all_modules[m_idx](_x)
        m_idx += 1

        for i in range(len(self.hidden_dim) - 1):
            _x = self.up(_x)
            _x = self.all_modules[m_idx](_x)
            m_idx += 1
            _x = self.all_modules[m_idx](torch.cat([_x,xs.pop()],dim=1))
            m_idx += 1
        
        assert not xs # xs has no item

        _x = self.all_modules[m_idx](_x)
        m_idx += 1

        assert m_idx == len(self.all_modules)
        _x = self.act(_x)

        return _x
    
    def get_timestep_embedding(self,timesteps, embedding_dim, max_positions=10000):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)   # magic number 10000 is from transformers
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb
        

        


if __name__ == '__main__':

    encoder = Encoder(9,10,4)
    decoder = Decoder(10,10,4,9)
    denoiser = Denoiser(320,10)

    encoder_fn = mutils.get_model_fn(encoder,train=False)
    decoder_fn = mutils.get_model_fn(decoder,train=False)
    denoiser_fn = mutils.get_model_fn(denoiser,train=False)
    
    temp_data = torch.randn((2,320,9))
    output  = encoder_fn(temp_data)
    print("Encoder Output: ",output.shape)

    labels = torch.rand(2)
    output = denoiser_fn(output.permute(0,2,1),labels)
    print("Denoiser Output: ",output.shape)


    z = torch.randn((output.permute(0,2,1).shape))
    output  = decoder_fn(z)
    print("Decoder Output: ",output.shape)
    
