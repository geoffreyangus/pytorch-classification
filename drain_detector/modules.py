import numpy as np
import torch.nn.functional as F
from . import networks
from .base_model import BaseModel
import util.util as util
from torch.autograd import Variable
from collections import OrderedDict
import os
import torch
import torch.nn as nn


class LinearDecoder(nn.Module):

    def __init__(self, num_classes, num_layers=2, dropout_p=0.0):
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


class SRCNN(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero', nclass=3):
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
