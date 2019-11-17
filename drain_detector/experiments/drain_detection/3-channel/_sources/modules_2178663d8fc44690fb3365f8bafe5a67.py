import os
import re
import functools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from torch.autograd import Variable


class LinearDecoder(nn.Module):

    def __init__(self, num_classes, num_layers=2, encoding_size=1024, dropout_p=0.0):
        """
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers - 1):
            self.layers.append(nn.Linear(in_features=encoding_size,
                                            out_features=encoding_size))
            self.layers.append(nn.Dropout(p=dropout_p))
        # classification layer
        self.layers.append(nn.Linear(in_features=encoding_size,
                                     out_features=num_classes))

    def forward(self, x):
        """
        """
        for layer in self.layers:
            x = layer(x)
        return x

    
class ClippedDenseNet(nn.Module):
    
    def __init__(self, n_classes=14, pretrained=True, weights_path=None):
        """
        """
        super().__init__()
        self.densenet121 = densenet121(pretrained=pretrained)
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(self.densenet121.classifier.in_features, n_classes),
            nn.Sigmoid()
        )
        # TODO: get this to work
        if weights_path:
            state_dict = torch.load(weights_path)['state_dict']
            state_dict = {re.sub(r'module\.densenet121', r'densenet121', k): v for k, v in state_dict.items()}
            state_dict = {re.sub(r'norm\.', r'norm', k): v for k, v  in state_dict.items()}
            state_dict = {re.sub(r'conv\.', r'conv', k): v for k, v  in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
            num_loaded = len(set(self.state_dict().keys()) & set(state_dict.keys()))
            num_total = len(state_dict.keys())
            if num_loaded < num_total:
                missing_params = set(state_dict.keys()).symmetric_difference(set(self.state_dict().keys()))
                logging.info("Could not load these parameters due to name mismatch: " + str(missing_params))
            logging.info(f"Loaded {num_loaded}/{num_total} pretrained parameters")
            exit()
        
        self.densenet121.classifier = nn.Identity()
        
    def forward(self, x):
        """
        """
        out = self.densenet121(x)
        return out
    

# requires strange input

class SRCNN(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero', nclass=3, weights_path=None):
		super(SRCNN, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.nclass = output_nc
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		# 512x512x3
		self.blk1 = nn.Sequential(nn.Conv2d(input_nc+output_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
                            norm_layer(ngf),
                            nn.ReLU(True),
                            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
		#512x512x64
		self.blk2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf*2),
                            nn.ReLU(True),
                            ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
		#256x256x128
		self.blk3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf*4),
                            nn.ReLU(True),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
		#128x128x256

		self.conv_lstm = CLSTMCell(input_dim=4*ngf,
                             hidden_dim=4*ngf,
                             kernel_size=(3, 3),
                             bias=use_bias
                             )

		#64x64x256
		self.blk5 = nn.Sequential(ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            nn.ConvTranspose2d(
                            	ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf*2),
                            nn.ReLU(True))
		#1280x128x256
		self.blk6 = nn.Sequential(ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            nn.ConvTranspose2d(
                            	ngf*4, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf),
                            nn.ReLU(True))
		#256x256x128
		self.blk7 = nn.Sequential(ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias),
                            nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias))

		self.up2h = nn.ConvTranspose2d(
			4*ngf, 4*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.up2c = nn.ConvTranspose2d(
			4*ngf, 4*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.up2i = nn.ConvTranspose2d(
			output_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

	def forward(self, input_list):
		batch_size, cc, hh, ww = input_list[2].size()

		h, c = self._init_hidden(batch_size=batch_size,
		                         spatial_size=(int(hh/16), int(ww/16)))

		seq_len = 3

		output = []

		I = Variable(torch.zeros(batch_size, self.nclass,
                           int(hh/4), int(ww/4))).cuda()
		#input list small-mid-large
		for t in range(seq_len):
			# print(I.data.cpu().size())
			# print(input_list[t].data.cpu().size())
			#h
			e1 = self.blk1(torch.cat((input_list[t], I), dim=1))
			#h
			e2 = self.blk2(e1)
			#h/2
			e3 = self.blk3(e2)
			# h/4
			h, c = self.conv_lstm(input_tensor=e3, cur_state=[h, c])
			# h/8
			d3 = self.blk5(h)

			# h/4
			d2 = self.blk6(torch.cat((d3, e2), 1))
			# h/2
			I = self.blk7(torch.cat((d2, e1), 1))

			# print(I.data.cpu().size())

			output.append(I)

			scale_f = 2

			h = self.up2h(h)
			c = self.up2c(c)
			I = self.up2i(I)

			I = torch.nn.functional.softmax(I, dim=1)*2-1
		return output

	def _init_hidden(self, batch_size, spatial_size):
		return self.conv_lstm.init_hidden(batch_size, spatial_size)

    
# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(
			dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                 norm_layer(dim),
                 nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                 norm_layer(dim)
                 ]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


class CLSTMCell(nn.Module):

	def __init__(self, input_dim, hidden_dim, kernel_size, bias):
		"""
		Initialize ConvLSTM cell.

		Parameters
		----------
		input_size: (int, int)
			Height and width of input tensor as (height, width).
		input_dim: int
			Number of channels of input tensor.
		hidden_dim: int
			Number of channels of hidden state.
		kernel_size: (int, int)
			Size of the convolutional kernel.
		bias: bool
			Whether or not to add the bias.
		"""

		super(CLSTMCell, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size = kernel_size
		self.padding = kernel_size[0] // 2, kernel_size[1] // 2
		self.bias = bias

		self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                        out_channels=4 * self.hidden_dim,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        bias=self.bias)

	def forward(self, input_tensor, cur_state):

		h_cur, c_cur = cur_state

		# concatenate along channel axis
		combined = torch.cat([input_tensor, h_cur], dim=1)

		combined_conv = self.conv(combined)
		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return h_next, c_next

	def init_hidden(self, batch_size, spatial_size):
		return (Variable(torch.zeros(batch_size, self.hidden_dim, spatial_size[0], spatial_size[1])).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, spatial_size[0], spatial_size[1])).cuda())

