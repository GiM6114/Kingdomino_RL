import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FC(nn.Module):
    def __init__(self, input_size, output_size, l, n, activation='SELU'):
        super().__init__()
        if isinstance(n, int):
            n = [n]*l # all layers same size
        prev_n = input_size
        layers = []
        for i in range(l):
            layers.append(nn.Linear(prev_n, n[i]))  # Hidden layers
            layers.append(getattr(nn, activation)())  # Activation function
            prev_n = n[i]
        layers.append(nn.Linear(prev_n, output_size))
        layers.append(nn.SELU())  # Activation function
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# can accept inputs of any size
class CNN(nn.Module):
    def __init__(self, input_channels, layers_params, activation='ReLU'):
        super().__init__()
        layers = []
        for l_p in layers_params:
            if l_p['type'] == 'conv':
                l_p['params']['in_channels'] = input_channels
                layers.append(nn.Conv2d(**l_p['params']))
                layers.append(getattr(nn, activation)())
                input_channels = l_p['params']['out_channels']
            elif l_p['type'] == 'maxpool':
                layers.append(nn.MaxPool2d(**l_p['params']))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    

# assuming square input
def compute_num_features(layers_params, expected_inpt_size):
    size = expected_inpt_size
    last_n_channels = -1
    for l_p in layers_params:
        l_pp = l_p['params']
        size = (size - l_pp['kernel_size'] + 2*l_pp['padding'] / l_pp['stride']) + 1
        if l_p['type'] == 'conv':
            last_n_channels = l_pp['out_channels']
    return int(size*size*last_n_channels)  

# restricts input to some size
# layers of FC after conv
class CNNFC(nn.Module):
    def __init__(self, input_channels, cnn_layers_params, fc_layers_params, expected_inpt_size):
        super().__init__()
        self.cnn = CNN(input_channels, cnn_layers_params)
        n_features_after_cnn = compute_num_features(
            cnn_layers_params, 
            expected_inpt_size)
        self.fc = FC(
            input_size=n_features_after_cnn,
            **fc_layers_params)
        
    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, 'b c w h -> b (c w h)')
        return self.fc(x)
    
class AdaptiveCNN(nn.Module):
    def __init__(
            self, aux_info_size, in_channels,
            conv_channels, conv_kernel_size, conv_stride, l,
            pool_place, pool_kernel_size, pool_stride,
            FMN_l, FMN_n):
        super().__init__()
        
        prev_n_channels = in_channels
        FMNs = []
        for i in range(l):
            parameters_size = (prev_n_channels*conv_channels[i]*(conv_kernel_size[i]**2))+conv_channels[i]
            FMNs.append(FC(aux_info_size, parameters_size, FMN_l, FMN_n))
            prev_n_channels = conv_channels[i]
        self.FMNs = nn.ModuleList(FMNs)
        
        self.conv_stride = conv_stride
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_channels = conv_channels
        self.pool_place = pool_place
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.l = l
        
    def forward(self, main, side):
        prev_n_channels = self.in_channels
        for i in range(self.l):
            wb = self.FMNs[i](side) # (batch,(kernel_size**2)*prev_n_channels*n_channels + n_channels)
            w = wb[:,:-self.conv_channels[i]].reshape(main.shape[0], self.conv_channels[i], prev_n_channels, self.conv_kernel_size[i], self.conv_kernel_size[i])
            b = wb[:,-self.conv_channels[i]:]
            pad = self.conv_kernel_size[i]//2
            original_main_shape = main.shape
            main = F.pad(main, (pad,pad,pad,pad), 'constant', -1) # pad each dimensions
            main = F.conv2d(
                input=main.reshape((1,-1)+main.shape[2:]), 
                weight=w.reshape((-1,)+w.shape[2:]),
                bias=b.reshape((-1,)+b.shape[2:]),
                groups=w.shape[0]).reshape((original_main_shape[0], w.shape[1])+original_main_shape[2:])
            if self.pool_place[i] != 0:
                main = F.max_pool2d(main, self.pool_kernel_size[i], self.pool_stride[i])
            main = F.selu(main)
            prev_n_channels = self.conv_channels[i]
        return main