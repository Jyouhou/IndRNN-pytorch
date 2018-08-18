import pdb
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.functional as f
import numpy as np


CUDA = torch.cuda.is_available()


class BatchNorm(nn.BatchNorm1d):
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1)  # uniform_()
            self.bias.data.zero_()


class IndRNNCell(nn.Module):
    """
    IndRNN Cell computes:

        $$h_t = \sigma(w_{ih} x_t + b_{ih}  +  w_{hh} (*) h_{(t-1)})$$

    \sigma is sigmoid or relu

    hyper-params:

        1. hidden_size
        2. input_size
        3. bias: true or false
        4. act: the nonlinearity function ("tanh", "relu", "sigmoid")
        5. hidden_min_abs & hidden_max_abs
        6. reccurent_only: only computes the reccurent part for faster computation.
        7. init: how to initialize the params. Default norm for N(0,1/\sqrt(size)); constant; uniform; orth
        8. gradient_clip: `(min,max)` or `bound`

    inputs:

        1. Input: (batch, input_size)
        2. Hidden: (batch, hidden_size)

        batch first by default

    output:

        1. output: (batch, hidden_size)
        1. hidden state: (batch, hidden_size)

    params:

        1. weight_ih: (hidden_size,input_size)
        2. weight_hh: (1,hidden_size)
        3. bias_ih: (1,hidden_size) or None

    usage:

        >>> cell = IndRNNCell(100,128)
        >>> Input = torch.randn(32,100)
        >>> Hidden = torch.randn(32,128)
        >>> _, h = cell(Input, Hidden)

    """

    def __init__(self, input_size, hidden_size, bias=True, act="relu",
                 hidden_min_abs=0, hidden_max_abs=2, reccurent_only=False,
                 gradient_clip=None, init_ih="norm",
                 input_weight_initializer=None,
                 recurrent_weight_initializer=None,
                 name="Default", debug=False):

        super(IndRNNCell, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.act = act
        self.reccurent_only = reccurent_only
        self.init_ih = init_ih
        self.input_weight_initializer = input_weight_initializer
        self.recurrent_weight_initializer = recurrent_weight_initializer
        self.name = name
        self.debug = debug
        if self.act is None:
            self.activation = F.tanh
        elif self.act == "relu":
            self.activation = F.relu
        elif self.act == "sigmoid":
            self.activation = F.sigmoid
        elif self.act == "tanh":
            self.activation = None
        else:
            raise RuntimeError(f"Unknown activation type: {self.nonlinearity}")
        if not self.reccurent_only:
            self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        else:
            self.register_parameter('weight_ih', None)
        self.weight_hh = Parameter(torch.Tensor(1, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(1, hidden_size))
        else:
            self.register_parameter('bias_ih', None)

        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                assert len(gradient_clip) == 2
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g
            if not self.reccurent_only:
                self.weight_ih.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_hh.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            if bias:
                self.bias_ih.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
        # debug
        # if self.debug:
        #    pdb.set_trace()
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "bias" in name:
                weight.data.zero_()
            elif "weight" in name:
                if self.input_weight_initializer and "weight_ih" in name:
                    self.input_weight_initializer(weight)
                elif self.recurrent_weight_initializer and "weight_hh" in name:
                    self.recurrent_weight_initializer(weight)
                elif "constant" in self.init_ih:
                    nn.init.constant_(weight, 1.0)
            else:
                weight.data.normal_(0, 0.01)
        self.clip_weight()

    def clip_weight(self):
        if self.hidden_min_abs:
            abs_kernel = torch.abs(self.weight_hh.data).clamp(min=self.hidden_min_abs)
            self.weight_hh.data = torch.sign(self.weight_hh.data) * abs_kernel
        if self.hidden_max_abs:
            self.weight_hh.data = self.weight_hh.clamp(min=-self.hidden_max_abs, max=self.hidden_max_abs)
        self.weight_hh.data.detach_()

    def forward(self, Input, Hidden):

        if not self.reccurent_only:
            h = F.linear(Input, self.weight_ih) + self.weight_hh * Hidden
            if self.bias:
                h += self.bias_ih
        else:
            h = Input + self.weight_hh * Hidden
        if self.activation:
            h = self.activation(h)
        return h, h


class BasicIndRNN(nn.Module):
    """
    the basic IndRNN architecture as described in :
        [Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR 2018.](https://arxiv.org/abs/1803.04831)
        Section4.1 (a)

    hyper-params:
        1. hidden_size
        2. input_size
        3. num_layer: number of layer
        4. bias: true or false
        5. act: the nonlinearity function ("tanh", "relu", "sigmoid")
        6. hidden_min_abs & hidden_max_abs
        7. init: how to initialize the params. Default norm for N(0,1/\sqrt(size)); constant; uniform; orth
        8. gradient_clip: `(min,max)` or `bound`
        9. batch_norm: True/False, apply batch_norm or not
        10. bidirection: True,False

    inputs:

        1. Input: (batch, time, input_size)
        2. h_0: (batch, hidden_size * num_direction)

        batch first by default

    output:

        1. output: (batch, seq_len, hidden_size * num_directions)
        2. h_n: (num_layers * num_directions, batch, hidden_size)

    attributes:
        1. cell: List()
        2. Wx: input weight--List(hidden_size, input_size)*num_layers

    """

    def __init__(self, input_size, hidden_size, num_layer=1, bidirection=False, bidirection_add=False,
                 bias=True, act="relu", batch_norm=True,
                 hidden_min_abs=0, hidden_max_abs=2,
                 gradient_clip=None, init_ih="uniform/-0.001/0.001",
                 input_weight_initializers=None,
                 recurrent_weight_initializers=None,
                 debug=False, batch_norm_momentum=0.9):
        super(BasicIndRNN, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.bidirection = bidirection
        self.bidirection_add = bidirection_add
        self.bias = bias
        self.act = act
        self.gradient_clip = gradient_clip
        self.init_ih = init_ih
        self.batch_norm = batch_norm
        self.num_direction = 2 if self.bidirection else 1
        self.input_weight_initializers = input_weight_initializers
        self.recurrent_weight_initializers = recurrent_weight_initializers
        self.debug = debug
        self.cells = []
        self.InputMapping = []
        for i in range(self.num_layer):
            direction_cells = []
            InputMappingDir = []
            Input_size = self.input_size if i == 0 else self.hidden_size * (self.num_direction - int(self.bidirection and self.bidirection_add))
            for Dir in range(self.num_direction):
                direction_cells.append(IndRNNCell(input_size=Input_size, hidden_size=self.hidden_size, bias=False, act=self.act,
                                                  hidden_min_abs=self.hidden_min_abs, hidden_max_abs=self.hidden_max_abs, reccurent_only=True,
                                                  gradient_clip=self.gradient_clip, init_ih=self.init_ih,
                                                  input_weight_initializer=self.input_weight_initializers[i] if self.input_weight_initializers else None,
                                                  recurrent_weight_initializer=self.recurrent_weight_initializers[i] if self.recurrent_weight_initializers else None,
                                                  name=f"Layer:{Dir}",
                                                  debug=debug))
                InputMappingDir.append(nn.Linear(Input_size, self.hidden_size, bias=True))
                nn.init.constant_(InputMappingDir[-1].bias, 0.)
                if "constant" in self.init_ih:
                    _, value = self.init_ih.split("/")
                    value = float(value)
                    nn.init.constant_(InputMappingDir[-1].weight, value)
                elif "norm" in self.init_ih:
                    _, mean, std = self.init_ih.split("/")
                    Size = np.sqrt(Input_size)
                    if mean == "size":
                        mean = 0
                    else:
                        mean = float(mean)
                    if std == "size":
                        std = 1 / Size
                    else:
                        std = float(std)
                    nn.init.normal_(InputMappingDir[-1].weight, mean, std)
                elif "uniform" in self.init_ih:
                    _, lower, upper = self.init_ih.split("/")
                    Size = np.sqrt(Input_size)
                    if lower == "size":
                        lower = -1 / Size
                    else:
                        lower = float(lower)
                    if upper == "size":
                        upper = 1 / Size
                    else:
                        upper = float(upper)
                    nn.init.uniform_(InputMappingDir[-1].weight, lower, upper)
                else:
                    nn.init.normal_(InputMappingDir[-1].weight, 0, 0.001)
            self.cells.append(nn.ModuleList(direction_cells))
            self.InputMapping.append(nn.ModuleList(InputMappingDir))
        self.cells = nn.ModuleList(self.cells)
        self.InputMapping = nn.ModuleList(self.InputMapping)

        if self.batch_norm:
            self.batch_norm_layers = []
            for i in range(self.num_layer):
                self.batch_norm_layers.append(BatchNorm(self.hidden_size * (self.num_direction - int(self.bidirection and self.bidirection_add)),
                                                        eps=1e-4,
                                                        momentum=batch_norm_momentum))
            self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)
        # if self.debug:
        #    pdb.set_trace()

    def forward(self, Input, h_0s=None):
        """
         inputs:

            1. Input: (batch, time, input_size)
            2. h_0: (batch, hidden_size * num_direction)

            batch first by default

        output:

            1. output: (batch, seq_len, hidden_size * (num_directions-int(self.bidirection_add)))
            2. h_n: (num_layers * (num_directions-int(self.bidirection_add)), batch, hidden_size)
        """
        batch_size = Input.size(0)
        time_len = Input.size(1)
        output_last_layer = Input  # (batch, time, input_size)
        hiddens = []
        for i, Layer in enumerate(self.cells):
            # the i-th layer
            if h_0s is None:
                h = torch.zeros(batch_size, self.hidden_size * (self.num_direction), device="cuda" if CUDA else "cpu")
            else:
                h = h_0s[i]
            outputs_dir = []
            hiddens_dir = []
            for Dir, cell in enumerate(Layer):
                h_cell = h[
                    :, self.hidden_size * Dir: self.hidden_size * (Dir + 1)]
                cell.clip_weight()
                mapped_input = self.InputMapping[i][Dir](output_last_layer)
                outputs = []
                for T in range(time_len):
                    if Dir == 1:
                        h_cell, _ = cell(mapped_input[:, time_len - T - 1, :], h_cell)
                    else:
                        h_cell, _ = cell(mapped_input[:, T, :], h_cell)
                    outputs.append(h_cell)
                if Dir == 1:
                    outputs = outputs[::-1]
                outputs_dir.append(torch.stack(outputs, 1))
                hiddens_dir.append(h_cell)
            if self.bidirection_add and self.bidirection:
                output_last_layer = (outputs_dir[0] + outputs_dir[1]) / 2  # average
                hiddens.append((hiddens_dir[0] + hiddens_dir[1]) / 2)
            else:
                output_last_layer = torch.cat(outputs_dir, -1)  # or concatenation
                hiddens.append(torch.cat(hiddens_dir, -1))

            if self.batch_norm:
                output_last_layer = self.batch_norm_layers[i](output_last_layer.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        return output_last_layer, torch.stack(hiddens, 0)


class BasicIndRNNLayerNorm(nn.Module):
    """
    the basic IndRNN architecture as described in :
        [Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR 2018.](https://arxiv.org/abs/1803.04831)
        Section4.1 (a)

    hyper-params:
        1. hidden_size
        2. input_size
        3. num_layer: number of layer
        4. bias: true or false
        5. act: the nonlinearity function ("tanh", "relu", "sigmoid")
        6. hidden_min_abs & hidden_max_abs
        7. init: how to initialize the params. Default norm for N(0,1/\sqrt(size)); constant; uniform; orth
        8. gradient_clip: `(min,max)` or `bound`
        9. batch_norm: True/False, apply batch_norm or not
        10. bidirection: True,False

    inputs:

        1. Input: (batch, time, input_size)
        2. h_0: (batch, hidden_size * num_direction)

        batch first by default

    output:

        1. output: (batch, seq_len, hidden_size * num_directions)
        2. h_n: (num_layers * num_directions, batch, hidden_size)

    attributes:
        1. cell: List()
        2. Wx: input weight--List(hidden_size, input_size)*num_layers

    """

    def __init__(self, input_size, hidden_size, num_layer=1, bidirection=False, bidirection_add=False,
                 bias=True, act="relu", layer_norm=True,
                 hidden_min_abs=0, hidden_max_abs=2,
                 gradient_clip=None, init_ih="uniform/-0.001/0.001",
                 input_weight_initializers=None,
                 recurrent_weight_initializers=None,
                 debug=False, batch_norm_momentum=0.9,
                 expected_time_len=784):
        super(BasicIndRNNLayerNorm, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.bidirection = bidirection
        self.bidirection_add = bidirection_add
        self.bias = bias
        self.act = act
        self.gradient_clip = gradient_clip
        self.init_ih = init_ih
        self.layer_norm = layer_norm
        self.num_direction = 2 if self.bidirection else 1
        self.input_weight_initializers = input_weight_initializers
        self.recurrent_weight_initializers = recurrent_weight_initializers
        self.expected_time_len = expected_time_len
        self.debug = debug
        self.cells = []
        self.InputMapping = []

        if self.act == "tanh":
            self.activation = F.tanh
        elif self.act == "relu":
            self.activation = F.relu
        elif self.act == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise RuntimeError(f"Unknown activation type: {self.nonlinearity}")

        for i in range(self.num_layer):
            direction_cells = []
            InputMappingDir = []
            Input_size = self.input_size if i == 0 else self.hidden_size * (self.num_direction - int(self.bidirection and self.bidirection_add))
            for Dir in range(self.num_direction):
                direction_cells.append(IndRNNCell(input_size=Input_size, hidden_size=self.hidden_size, bias=False, act=None,
                                                  hidden_min_abs=self.hidden_min_abs, hidden_max_abs=self.hidden_max_abs, reccurent_only=True,
                                                  gradient_clip=self.gradient_clip, init_ih=self.init_ih,
                                                  input_weight_initializer=self.input_weight_initializers[i] if self.input_weight_initializers else None,
                                                  recurrent_weight_initializer=self.recurrent_weight_initializers[i] if self.recurrent_weight_initializers else None,
                                                  name=f"Layer:{Dir}",
                                                  debug=debug))
                InputMappingDir.append(nn.Linear(Input_size, self.hidden_size, bias=True))
                nn.init.constant_(InputMappingDir[-1].bias, 0.)
                if "constant" in self.init_ih:
                    _, value = self.init_ih.split("/")
                    value = float(value)
                    nn.init.constant_(InputMappingDir[-1].weight, value)
                elif "norm" in self.init_ih:
                    _, mean, std = self.init_ih.split("/")
                    Size = np.sqrt(Input_size)
                    if mean == "size":
                        mean = 0
                    else:
                        mean = float(mean)
                    if std == "size":
                        std = 1 / Size
                    else:
                        std = float(std)
                    nn.init.normal_(InputMappingDir[-1].weight, mean, std)
                elif "uniform" in self.init_ih:
                    _, lower, upper = self.init_ih.split("/")
                    Size = np.sqrt(Input_size)
                    if lower == "size":
                        lower = -1 / Size
                    else:
                        lower = float(lower)
                    if upper == "size":
                        upper = 1 / Size
                    else:
                        upper = float(upper)
                    nn.init.uniform_(InputMappingDir[-1].weight, lower, upper)
                else:
                    nn.init.normal_(InputMappingDir[-1].weight, 0, 0.001)
            self.cells.append(nn.ModuleList(direction_cells))
            self.InputMapping.append(nn.ModuleList(InputMappingDir))
        self.cells = nn.ModuleList(self.cells)
        self.InputMapping = nn.ModuleList(self.InputMapping)

        if self.layer_norm:
            self.layer_norm_layers = []
            for i in range(self.num_layer):
                self.layer_norm_layers.append(nn.LayerNorm([  # self.expected_time_len,
                    self.hidden_size * (self.num_direction - int(self.bidirection and self.bidirection_add))],
                    eps=1e-4))
            self.layer_norm_layers = nn.ModuleList(self.layer_norm_layers)
        # if self.debug:
        #    pdb.set_trace()

    def forward(self, Input, h_0s=None):
        """
         inputs:

            1. Input: (batch, time, input_size)
            2. h_0: (batch, hidden_size * num_direction)

            batch first by default

        output:

            1. output: (batch, seq_len, hidden_size * (num_directions-int(self.bidirection_add)))
            2. h_n: (num_layers * (num_directions-int(self.bidirection_add)), batch, hidden_size)
        """
        batch_size = Input.size(0)
        time_len = Input.size(1)
        output_last_layer = Input  # (batch, time, input_size)
        hiddens = []
        for i, Layer in enumerate(self.cells):
            # the i-th layer
            if h_0s is None:
                h = torch.zeros(batch_size, self.hidden_size * (self.num_direction), device="cuda" if CUDA else "cpu")
            else:
                h = h_0s[i]
            outputs_dir = []
            hiddens_dir = []
            for Dir, cell in enumerate(Layer):
                h_cell = h[
                    :, self.hidden_size * Dir: self.hidden_size * (Dir + 1)]
                cell.clip_weight()
                mapped_input = self.InputMapping[i][Dir](output_last_layer)
                outputs = []
                for T in range(time_len):
                    if Dir == 1:
                        h_cell, _ = cell(mapped_input[:, time_len - T - 1, :], h_cell)
                    else:
                        h_cell, _ = cell(mapped_input[:, T, :], h_cell)
                    outputs.append(h_cell)
                if Dir == 1:
                    outputs = outputs[::-1]
                outputs_dir.append(torch.stack(outputs, 1))
                hiddens_dir.append(h_cell)
            if self.bidirection_add and self.bidirection:
                output_last_layer = (outputs_dir[0] + outputs_dir[1]) / 2  # average
                hiddens.append((hiddens_dir[0] + hiddens_dir[1]) / 2)
            else:
                output_last_layer = torch.cat(outputs_dir, -1)  # or concatenation
                hiddens.append(torch.cat(hiddens_dir, -1))

            if self.layer_norm:
                output_last_layer = self.layer_norm_layers[i](output_last_layer)
                output_last_layer = self.activation(output_last_layer)
        return output_last_layer, torch.stack(hiddens, 0)


class ResIndRNN(nn.Module):
    """
    the residual IndRNN architecture as described in :
        [Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR 2018.](https://arxiv.org/abs/1803.04831)
        Section4.1 (b)

    hyper-params:
        1. hidden_size
        2. input_size
        3. num_layer: number of layer
        4. bias: true or false
        5. act: the nonlinearity function ("tanh", "relu", "sigmoid")
        6. hidden_min_abs & hidden_max_abs
        7. init: how to initialize the params. Default norm for N(0,1/\sqrt(size)); constant; uniform; orth
        8. gradient_clip: `(min,max)` or `bound`
        9. batch_norm: True/False, apply batch_norm or not
        10. bidirection: True,False
        11. dropout_rate
        12. res_block_size
        13. output_class_num

    inputs:

        1. Input: (batch, time, input_size)
        2. h_0: (batch, hidden_size * num_direction)

        batch first by default

    output:

        1. output: (batch, seq_len, hidden_size * num_directions)
        2. h_n: (num_layers * num_directions, batch, hidden_size)

    attributes:
        1. cell: List()
        2. Wx: input weight--List(hidden_size, input_size)*num_layers

    """

    def __init__(self, input_size, hidden_size=2000, num_layer=11, bidirection=False, bidirection_add=False,
                 bias=True, act="relu", batch_norm=True,
                 hidden_min_abs=0, hidden_max_abs=2,
                 gradient_clip=None, init_ih="uniform/-0.001/0.001",
                 input_weight_initializers=None,
                 recurrent_weight_initializers=None,
                 debug=False, batch_norm_momentum=0.9,
                 dropout_rate=0.3, residual_layers=2,
                 expected_time_step=50):
        super(ResIndRNN, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.bidirection = bidirection
        self.bidirection_add = bidirection_add
        self.bias = bias
        self.act = act
        self.gradient_clip = gradient_clip
        self.init_ih = init_ih
        self.batch_norm = batch_norm
        self.num_direction = 2 if self.bidirection else 1
        self.input_weight_initializers = input_weight_initializers
        self.recurrent_weight_initializers = recurrent_weight_initializers
        self.debug = debug
        self.dropout_rate = dropout_rate
        self.residual_layers = residual_layers
        self.expected_time_step = expected_time_step

        # cells
        self.cells = []
        self.InputMapping = []
        for i in range(self.num_layer):
            direction_cells = []
            InputMappingDir = []
            Input_size = self.input_size if i == 0 else self.hidden_size * (
                self.num_direction - int(self.bidirection and self.bidirection_add))
            for Dir in range(self.num_direction):
                direction_cells.append(
                    IndRNNCell(input_size=Input_size, hidden_size=self.hidden_size, bias=False, act=self.act,
                               hidden_min_abs=self.hidden_min_abs, hidden_max_abs=self.hidden_max_abs,
                               reccurent_only=True,
                               gradient_clip=self.gradient_clip, init_ih=self.init_ih,
                               input_weight_initializer=self.input_weight_initializers[i] if self.input_weight_initializers else None,
                               recurrent_weight_initializer=self.recurrent_weight_initializers[i] if self.recurrent_weight_initializers else None,
                               name=f"Layer:{Dir}",
                               debug=debug))
                InputMappingDir.append(nn.Linear(Input_size, self.hidden_size, bias=True))
            self.cells.append(nn.ModuleList(direction_cells))
            self.InputMapping.append(nn.ModuleList(InputMappingDir))
        self.cells = nn.ModuleList(self.cells)
        self.InputMapping = nn.ModuleList(self.InputMapping)

        # BNs
        if self.batch_norm:
            self.batch_norm_layers = []
            for i in range(self.num_layer):
                batch_norm_layer = []
                for i in range(expected_time_step):
                    batch_norm_layer_dir = []
                    for Dir in range(self.num_direction):
                        batch_norm_layer_dir.append(BatchNorm(
                            self.hidden_size * (self.num_direction - int(self.bidirection and self.bidirection_add)),
                            eps=1e-4,
                            momentum=batch_norm_momentum))
                        if i > 0:
                            # TODO default for framewise?
                            batch_norm_layer_dir[Dir].weight = batch_norm_layer[0][Dir].weight
                            batch_norm_layer_dir[Dir].bias = batch_norm_layer[0][Dir].bias
                    batch_norm_layer.append(nn.ModuleList(batch_norm_layer_dir))

                self.batch_norm_layers.append(nn.ModuleList(batch_norm_layer))
            self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)

        # Dropout:
        self.Dropouts = []
        for i in range(self.num_layer):
            self.Dropouts.append(nn.Dropout(self.dropout_rate))
        # if self.debug:
        #    pdb.set_trace()
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "cells" in name:
                pass
            if "InputMapping" in name:
                if "weight" in name:
                    torch.nn.init.kaiming_uniform_(param.data, a=2, mode='fan_in')
            if 'batch_norm_layers' in name and 'weight' in name:
                param.data.fill_(1)
            if 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, Input, h_0s=None):
        """
         inputs:

            1. Input: (batch, time, input_size)
            2. h_0s: List[(batch, hidden_size * num_direction)] * num_layer

            batch first by default

        output:

            1. output: (batch, seq_len, hidden_size * (num_directions-int(self.bidirection_add)))
            2. h_n: (num_layers * (num_directions-int(self.bidirection_add)), batch, hidden_size)
        """
        batch_size = Input.size(0)
        time_len = Input.size(1)
        assert time_len == self.expected_time_step
        output_last_layer = Input  # (batch, time, input_size)
        last_mapped_input = []
        hiddens = []

        for i, Layer in enumerate(self.cells):
            # the i-th layer
            if h_0s is None:
                h = torch.zeros(batch_size, self.hidden_size * (self.num_direction),
                                device="cuda" if CUDA else "cpu")
            else:
                h = h_0s[i]
            outputs_dir = []
            hiddens_dir = []
            for Dir, cell in enumerate(Layer):
                h_cell = h[
                    :, self.hidden_size * Dir: self.hidden_size * (Dir + 1)]
                cell.clip_weight()

                if self.residual_layers > 0 and i >= self.residual_layers and i % self.residual_layers == 0:
                    assert len(last_mapped_input) > 0
                    mapped_input = self.InputMapping[i][Dir](output_last_layer) + last_mapped_input[-1]
                    last_mapped_input.append(mapped_input)
                elif i == 0:
                    mapped_input = self.InputMapping[i][Dir](output_last_layer)
                    last_mapped_input.append(mapped_input)
                else:
                    mapped_input = self.InputMapping[i][Dir](output_last_layer)  # D0

                outputs = []
                for T in range(time_len):
                    if self.batch_norm:
                        if Dir == 1:
                            h_cell, _ = cell(
                                self.batch_norm_layers[i][T][Dir](mapped_input[:, time_len - T - 1, :].contiguous()),
                                h_cell)
                        else:
                            h_cell, _ = cell(self.batch_norm_layers[i][T][Dir](mapped_input[:, T, :].contiguous()),
                                             h_cell)
                    else:
                        if Dir == 1:
                            h_cell, _ = cell(mapped_input[:, time_len - T - 1, :], h_cell)
                        else:
                            h_cell, _ = cell(mapped_input[:, T, :], h_cell)
                    outputs.append(h_cell)
                if Dir == 1:
                    outputs = outputs[::-1]
                outputs_dir.append(torch.stack(outputs, 1))
                hiddens_dir.append(h_cell)
            if self.bidirection_add and self.bidirection:
                output_last_layer = (outputs_dir[0] + outputs_dir[1]) / 2  # average
            else:
                output_last_layer = torch.cat(outputs_dir, -1)  # or concatenation
            hiddens.append(torch.cat(hiddens_dir, -1))  # TODO bidirection + add?
            output_last_layer = self.Dropouts[i](output_last_layer)

        return output_last_layer, hiddens


if __name__ == "__main__":
    # test part 1
    rnn = BasicIndRNN(16, 128, 3, True, act="relu", batch_norm=True, bidirection_add=True)
    input = torch.randn(32, 5, 16)
    output, h = rnn(input)
    # test part 2
    rnn = ResIndRNN(10, 32, 11, expected_time_step=5)
    input = torch.randn(4, 5, 10)
    output, h = rnn(input)
