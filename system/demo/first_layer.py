import torch
import torch.nn as nn
import torchvision.models as models
import sys
sys.path.append('/home/hwlee/senier/system')
from SEAL_Python.examples.seal import *

import numpy as np



# Define the convolutional layer function
class ConvLayer():
    # def __init__(self, input_shape, num_filters, kernel_size, stride, padding, weight, bias, position):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, weight, position, evaluator):
        super(ConvLayer, self).__init__()
        self.batch_size, self.input_height, self.input_width, self.input_channels = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = weight.transpose(3, 2, 1, 0)
        
        self.position = position
        
        self.evaluator = evaluator
    
    def conv_forward(self, input_volume):
        print(self.kernel.shape)
        self.kernel = np.round(self.kernel * 1e3).astype(int)
        self.kernel[self.kernel <= 0] = 1
        # print(self.kernel)
        
        output_height = (self.input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (self.input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        input_volume = np.pad(input_volume, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        self.position = np.pad(np.array(self.position), ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        print(self.position.shape)
        self.position = self.position.tolist()

        # output_volume = np.zeros((self.batch_size, output_height, output_width, self.num_filters)).tolist()
        output_volume = np.zeros((self.batch_size, self.num_filters, output_width, output_height)).tolist()
        
        output_position = []
        for i in range(output_height):
            output_position.append([0] * output_width)
        from tqdm import tqdm
        for i in tqdm(range(output_height)):
            for j in range(output_width):
                for f in range(self.num_filters):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    receptive_field = input_volume[:, h_start:h_end, w_start:w_end, :]
                    
                    if sum(sum(row[w_start:w_end]) for row in self.position[h_start:h_end]) != 0:
                        output_position[i][j] = 1
                        for a in range(self.batch_size):
                            temp = 0
                            temp_encryption = 0
                            temp_count = 0
                            for b in range(self.kernel_size):
                                for c in range(self.kernel_size):
                                    for d in range(self.input_channels):
                                        if self.position[h_start+b][w_start+c] == 0:
                                            temp += receptive_field[a,b,c,d] * self.kernel[b,c,d,f]
                                        else:
                                            if temp_count == 0:
                                                temp_mul = self.evaluator.multiply_plain(receptive_field[a,b,c,d], Plaintext(str(self.kernel[b,c,d,f])))
                                                temp_encryption = self.evaluator.add_plain(temp_mul, Plaintext(str(temp_encryption)))
                                                temp_count += 1
                                            else:
                                                temp_mul = self.evaluator.multiply_plain(receptive_field[a,b,c,d], Plaintext(str(self.kernel[b,c,d,f])))
                                                temp_encryption = self.evaluator.add(temp_mul, temp_encryption)
                            output_volume[a][f][j][i] = self.evaluator.add_plain(temp_encryption, Plaintext(str(temp)))
                    else:
                        output_volume[0][f][j][i] = np.sum(receptive_field * self.kernel[:, :, :, f], axis=(1, 2, 3))
        
        # return output_volume.transpose(0, 3, 2, 1), output_position
        return output_volume, output_position
    
class First_layer():
    def __init__(self, position, evaluator, batch_size=1):
        self.batch_size = batch_size
        net = models.resnext101_32x8d(pretrained=False)
        net_num_features = net.fc.in_features
        net.fc = nn.Linear(net_num_features, 17)

        net.load_state_dict(torch.load("../../best_model_final.pth"))

        conv1_weight = net.conv1.weight
        
        self.conv1 = ConvLayer((batch_size, 227, 227, 3), 64, 7, 2, 3, conv1_weight.detach().numpy(), position, evaluator)
        
    def forward(self, x):
        x, output_position = self.conv1.conv_forward(x)
        
        return x, output_position