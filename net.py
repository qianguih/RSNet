import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

from layers.slice_pool_layer.slice_pool_layer import *
from layers.slice_unpool_layer.slice_unpool_layer import *




class RSNet(nn.Module):
    def __init__(self, pool_type, num_slice=[None, None, None]):
        super(RSNet, self).__init__()
        # input: B, 1, N, 3
        
        #-- conv block 1
        self.conv_1 = nn.Conv2d( 1, 64, kernel_size=(1,9), stride=(1,1)  )
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d( 64, 64, kernel_size=(1,1), stride=(1,1)  )
        self.bn_2 = nn.BatchNorm2d(64)
        
        self.conv_3 = nn.Conv2d( 64, 64, kernel_size=(1,1), stride=(1,1)  )
        self.bn_3 = nn.BatchNorm2d(64)
        
        #-- RNN block
        num_slice_x, num_slice_y, num_slice_z = num_slice
        self.pool_x = SP(pool_type, num_slice_x)
        self.pool_y = SP(pool_type, num_slice_y)
        self.pool_z = SP(pool_type, num_slice_z)
        
        self.rnn_type = 'GRU'
        self.rnn_hidden_sz_list = [256, 128, 64, 64, 128, 256]
        
        self.rnn_x_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
        self.rnn_x_2 = nn.GRU(512, self.rnn_hidden_sz_list[1], 1, bidirectional=True)
        self.rnn_x_3 = nn.GRU(256, self.rnn_hidden_sz_list[2], 1, bidirectional=True)
        self.rnn_x_4 = nn.GRU(128, self.rnn_hidden_sz_list[3], 1, bidirectional=True)
        self.rnn_x_5 = nn.GRU(128, self.rnn_hidden_sz_list[4], 1, bidirectional=True)
        self.rnn_x_6 = nn.GRU(256, self.rnn_hidden_sz_list[5], 1, bidirectional=True)
                
        
        self.rnn_y_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
        self.rnn_y_2 = nn.GRU(512, self.rnn_hidden_sz_list[1], 1, bidirectional=True)
        self.rnn_y_3 = nn.GRU(256, self.rnn_hidden_sz_list[2], 1, bidirectional=True)
        self.rnn_y_4 = nn.GRU(128, self.rnn_hidden_sz_list[3], 1, bidirectional=True)
        self.rnn_y_5 = nn.GRU(128, self.rnn_hidden_sz_list[4], 1, bidirectional=True)
        self.rnn_y_6 = nn.GRU(256, self.rnn_hidden_sz_list[5], 1, bidirectional=True)
        
        self.rnn_z_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
        self.rnn_z_2 = nn.GRU(512, self.rnn_hidden_sz_list[1], 1, bidirectional=True)
        self.rnn_z_3 = nn.GRU(256, self.rnn_hidden_sz_list[2], 1, bidirectional=True)
        self.rnn_z_4 = nn.GRU(128, self.rnn_hidden_sz_list[3], 1, bidirectional=True)
        self.rnn_z_5 = nn.GRU(128, self.rnn_hidden_sz_list[4], 1, bidirectional=True)
        self.rnn_z_6 = nn.GRU(256, self.rnn_hidden_sz_list[5], 1, bidirectional=True)
        
        #-- conv block 3
        self.un_pool_x = SU()
        self.un_pool_y = SU()
        self.un_pool_z = SU()
                
        self.conv_6 = nn.Conv2d( 512, 512, kernel_size=(1,1), stride=(1,1)  )
        self.bn_6 = nn.BatchNorm2d(512)
        
        self.conv_7 = nn.Conv2d( 512, 256, kernel_size=(1,1), stride=(1,1)  )
        self.bn_7 = nn.BatchNorm2d(256)
        
        self.dp = nn.Dropout(p=0.3)
        
        self.conv_8 = nn.Conv2d( 256, 13, kernel_size=(1,1), stride=(1,1)  )
        
        self.relu = nn.ReLU(inplace=True)
    
        self._initialize_weights()


    def forward(self, x, x_slice_idx, y_slice_idx, z_slice_idx, hidden_list):
        
        num_batch, _, num_points, _ = x.size()
        
        x_hidden_1, x_hidden_2, x_hidden_3, x_hidden_4, x_hidden_5, x_hidden_6, y_hidden_1, y_hidden_2, y_hidden_3, y_hidden_4, y_hidden_5, y_hidden_6, z_hidden_1, z_hidden_2, z_hidden_3, z_hidden_4, z_hidden_5, z_hidden_6 = hidden_list
        
        #-- conv block 1
        conv_1 =  self.relu(  self.bn_1( self.conv_1(x) )  )  # num_batch, 64, num_points, 1
        conv_2 =  self.relu(  self.bn_2( self.conv_2(conv_1) )  ) # num_batch, 64, num_points, 1
        conv_3 =  self.relu(  self.bn_3( self.conv_3(conv_2) )  ) # num_batch, 64, num_points, 1
        
        #-- RNN block
        x_pooled = self.pool_x( conv_3, x_slice_idx )  # num_batch, 64, numSlices, 1
        y_pooled = self.pool_y( conv_3, y_slice_idx )
        z_pooled = self.pool_z( conv_3, z_slice_idx )
                
        x_pooled = x_pooled[:,:,:,0].permute( 2, 0, 1 ).contiguous()
        y_pooled = y_pooled[:,:,:,0].permute( 2, 0, 1 ).contiguous()
        z_pooled = z_pooled[:,:,:,0].permute( 2, 0, 1 ).contiguous()
        
        x_rnn_1, _ = self.rnn_x_1( x_pooled, x_hidden_1 )
        x_rnn_2, _ = self.rnn_x_2( x_rnn_1, x_hidden_2 )
        x_rnn_3, _ = self.rnn_x_3( x_rnn_2, x_hidden_3 )
        x_rnn_4, _ = self.rnn_x_4( x_rnn_3, x_hidden_4 )
        x_rnn_5, _ = self.rnn_x_5( x_rnn_4, x_hidden_5 )
        x_rnn_6, _ = self.rnn_x_6( x_rnn_5, x_hidden_6 )
        
        y_rnn_1, _ = self.rnn_y_1( y_pooled, y_hidden_1 )
        y_rnn_2, _ = self.rnn_y_2( y_rnn_1, y_hidden_2 )
        y_rnn_3, _ = self.rnn_y_3( y_rnn_2, y_hidden_3 )
        y_rnn_4, _ = self.rnn_y_4( y_rnn_3, y_hidden_4 )
        y_rnn_5, _ = self.rnn_y_5( y_rnn_4, y_hidden_5 )
        y_rnn_6, _ = self.rnn_y_6( y_rnn_5, y_hidden_6 )
        
        z_rnn_1, _ = self.rnn_z_1( z_pooled, z_hidden_1 )
        z_rnn_2, _ = self.rnn_z_2( z_rnn_1, z_hidden_2 )
        z_rnn_3, _ = self.rnn_z_3( z_rnn_2, z_hidden_3 )
        z_rnn_4, _ = self.rnn_z_4( z_rnn_3, z_hidden_4 )
        z_rnn_5, _ = self.rnn_z_5( z_rnn_4, z_hidden_5 )
        z_rnn_6, _ = self.rnn_z_6( z_rnn_5, z_hidden_6 )
        
        #-- uppooling
        x_rnn_6 = x_rnn_6.permute( 1, 2, 0 ).contiguous()
        x_rnn_6 = x_rnn_6.view( x_rnn_6.size(0), x_rnn_6.size(1), x_rnn_6.size(2), 1 )
        
        y_rnn_6 = y_rnn_6.permute( 1, 2, 0 ).contiguous()
        y_rnn_6 = y_rnn_6.view( y_rnn_6.size(0), y_rnn_6.size(1), y_rnn_6.size(2), 1 )
        
        z_rnn_6 = z_rnn_6.permute( 1, 2, 0 ).contiguous()
        z_rnn_6 = z_rnn_6.view( z_rnn_6.size(0), z_rnn_6.size(1), z_rnn_6.size(2), 1 )
        
        x_rnn_6 = self.un_pool_x( x_rnn_6, x_slice_idx )
        y_rnn_6 = self.un_pool_y( y_rnn_6, y_slice_idx )
        z_rnn_6 = self.un_pool_z( z_rnn_6, z_slice_idx )
        
        #-- conv block 3
        rnn = x_rnn_6 + y_rnn_6 + z_rnn_6
        
        conv_6 =  self.relu(  self.bn_6( self.conv_6(rnn) )  ) # num_batch, 512, num_points, 1
        conv_7 =  self.relu(  self.bn_7( self.conv_7(conv_6) )  ) # num_batch, 256, num_points, 1
        droped = self.dp(conv_7)
        conv_8 =   self.conv_8(droped)
        
        return conv_8
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def init_hidden(self, bsz = 1):
        weight = next(self.parameters()).data
        hidden_list = [ ]
        for i in range(3):
            for hid_sz in self.rnn_hidden_sz_list:
                if self.rnn_type == 'LSTM':
                    hidden_list.append(  (Variable(weight.new(2, bsz, hid_sz).zero_()),
                                          Variable(weight.new(2, bsz, hid_sz).zero_()))  )
                else:
                    hidden_list.append( Variable(weight.new(2, bsz, hid_sz).zero_()) )

        return hidden_list




























