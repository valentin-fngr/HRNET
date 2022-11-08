import torch.nn as nn 
import torchvision 
import numpy as np 
from collections import OrderedDict



class ConvBnRelu(nn.Module): 

    """
    The basic Conv > BatchNorm > Relu Block 

    Attributes
    ----------
    nb_channels : int
        number of input and output channels 
    """

    def __init__(self, nb_channels): 

        super().__init__()
        self.nb_channels = nb_channels
        
        self.conv = nn.Conv2d(nb_channels, nb_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(nb_channels)
        self.relu = nn.ReLU()

    def forward(self, x): 
        """
        Forward pass 
        """

        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class Stage1_hrnet(nn.Module): 
    """
    Fist stage of the hrnet model

    Attributes 
    ----------
    bottle_neck_channels : int 
        Mentioned in the paper, the first stage uses bottleneck layers with 64 channels
    nb_blocks: int 
        The number of blocks to add sequentially to compose a single stage  
    nb_channels : int 
        The number of channels to keep for the high resolution path (named C in the official paper)

    """
    def __init__(self, bottle_neck_channels, nb_channels, nb_blocks): 

        super().__init__()
        self.bottle_neck_channels = bottle_neck_channels
        self.nb_channels = nb_channels 
        self.nb_blocks = nb_blocks

        trunc = []
        for i in range(nb_blocks): 
            trunc.extend([
                nn.Conv2d(nb_channels, bottle_neck_channels, 3, 1, 1),
                nn.Conv2d(bottle_neck_channels, nb_channels, 1, 1, 0)
            ])

        self.trunc = nn.Sequential(*trunc)
        
        
    def forward(self, x): 
        """
        Forward pass for stage 1
        """
        out = self.trunc(x)
        return out



class BasicNetwork(nn.Module): 
    """
        A basic convolutiona: feed forward network 

    Attributes 
    ----------
    nb_blocks : int 
        The number of blocks to add sequentially to compose a single stage 
    nb_channels : int 
        The number of channels to keep for the high resolution path (named C in the official paper and should be stage_idx * C)
    """

    def __init__(self, nb_blocks, nb_channels): 

        super().__init__()
        self.nb_blocks = nb_blocks
        self.nb_channels = nb_channels

        trunc = [ConvBnRelu(nb_channels) for i in range(nb_blocks)]
        self.trunc = nn.Sequential(*trunc)

    def forward(self, x): 
        out = self.trunc(x)
        return out



class FusionModule(nn.Module): 
    """
    The FusionModule class

    Attributes 
    ----------

    nb_input : int 
        number of input tensor
    nb_channels: int 
        number of channels in the high resolution branch
    """

    def __init__(self, nb_input, nb_channels, add_branch=True):

        super().__init__()
        self.nb_input = nb_input
        self.nb_channels = nb_channels
        nb_output = nb_input + 1 if add_branch else nb_input
        self.nb_output = nb_output

        for i in range(nb_input):
            c_in = nb_channels * (1+i)
            for j in range(nb_output): 

                if i == j: 
                    self.add_module(f"keep_stage_{i+1}_to_stage_{j+1}", nn.Conv2d(c_in, c_in, 3, 1, 1))
                elif i < j: 

                    # 3x3 stride = 2 (x2*(j - i)) times
                    # handle the case when we have to downsample multiple times
                    # j = 2 
                    # i = 0 
                    # we need to downsample 2x 
                    # conv()
                    trunc = [nn.Conv2d(c_in + nb_channels*k,c_in + nb_channels*(k+1), 3, 2, 1) for k in range(0, j-i)]
                    self.add_module(f"downsample_from_stage_{i+1}_to_stage_{j+1}", nn.Sequential(*trunc))
                     
                else: 
                    # 1 x 1 bilinear > 1x1 convolution to match nb channels | be careful : use different scale_factor !!!!
                    scale_factor = 2**(i - j)
                    # print(f"Building upsampling from : {(i, j)} | expecting {2**i * nb_channels} to {2**j * nb_channels}")
                    print("from to : ", c_in, nb_channels * (j+1))
                    trunc = [
                        nn.Upsample(scale_factor=scale_factor, mode='bilinear'), 
                        nn.Conv2d(c_in, nb_channels * (j+1), 1, 1, 0)
                    ]
                    self.add_module(f"upsample_frmo_stage_{i+1}_to_stage_{j+1}", nn.Sequential(*trunc))

    def forward(self, *args): 
        """
        Forward pass 

        Parameters 
        ----------

        args : list(tensor)
            A list of tensors with different number of channels. These tensors are the input tensors 


        Returns
        -------
        output  : list(tensor) 
            A list of (stage + 1) output tensors with different number of channels  
        """

        if len(*args) != self.nb_input: 
            raise ValueError("Number of provided input is different that the number of input for this Fusion Module class")

        output = [None] * (self.nb_output)
        

        for i in range(self.nb_input):
            c_in = self.nb_channels * (1+i)
            for j in range(self.nb_output): 
                if i == j: 
                    # print("ff : ", j+1, i+1)
                    c_in = c_out = self.nb_channels * (i+1)
                    out_j = self._modules[f"keep_stage_{i+1}_to_stage_{j+1}"](list(*args)[i])
            
                elif i < j: 
                    # print("downsample from : ", i+1, j+1)
                    c_in = self.nb_channels * (i+1)
                    out_j = self._modules[f"downsample_from_stage_{i+1}_to_stage_{j+1}"](list(*args)[i])
                else: 
                    # 1 x 1 bilinear > 1x1 convolution to match nb channels | be careful : use different scale_factor !!!!
                    scale_factor = (i - j) * 2 
                    # print("upsample from : ", i+1, j+1, scale_factor)
                    # print("upsample input : ", list(*args)[i].shape)
                    # print("layer : ", self.layers[f"upsample_{i+1}_{j+1}"])
                    print("From to : ", i+1, j+1, scale_factor)
                    print("c_in ", c_in)
                    out_j = self._modules[f"upsample_frmo_stage_{i+1}_to_stage_{j+1}"](list(*args)[i])


                output[j] = out_j if output[j] is None else (output[j] + out_j)
        
        return output


        
class HRNETV2(nn.Module): 
    """
    The Hrnet_v2 model class

    Attributes 
    ----------
    nb_stages : int 
        The number of parallel networks and the number of stages (number of stages = number of parallel networks)
    nb_blocks : int 
        The number of blocks to add sequentially to compose a single stage 
    nb_channels : int 
        The number of channels to keep for the high resolution path (named C in the official paper)
    bottle_neck_channels : int
        Number of channels to use in the bottleneck convolution layer inside the stage 1 

    """
    def __init__(self, nb_stages, nb_blocks, nb_channels, bottle_neck_channels=64): 

        super().__init__()
        self.nb_stages = nb_stages 
        self.nb_blocks = nb_blocks 
        self.nb_channels = nb_channels
        self.bottle_neck_channels = bottle_neck_channels

        self.downsample_1 = nn.Conv2d(3, nb_channels, 3, 2, 1)
        self.downsample_2 = nn.Conv2d(nb_channels, nb_channels, 3, 2, 1)

        self.stage_1_1 = Stage1_hrnet(bottle_neck_channels, nb_channels, nb_blocks)
        self.stage_1_2 = BasicNetwork(nb_blocks, nb_channels)
        self.stage_1_3 = BasicNetwork(nb_blocks, nb_channels)
        self.stage_1_4 = BasicNetwork(nb_blocks, nb_channels)
        self.fusion_stage_1 = FusionModule(1, nb_channels)
        # TODO : 
        # make this flexible in case we add more stages !
        self.stage_2_1 = BasicNetwork(nb_blocks, 2*nb_channels)
        self.stage_2_2 = BasicNetwork(nb_blocks, 2*nb_channels)
        self.stage_2_3 = BasicNetwork(nb_blocks, 2*nb_channels)
        self.fusion_stage_2 = FusionModule(2, nb_channels)

        self.stage_3_1 = BasicNetwork(nb_blocks, 3*nb_channels)
        self.stage_3_2 = BasicNetwork(nb_blocks, 3*nb_channels)
        self.fusion_stage_3 = FusionModule(3, nb_channels)

        self.stage_4_1 = BasicNetwork(nb_blocks, 4*nb_channels)
        self.fusion_stage_4 = FusionModule(4,nb_channels, add_branch=False)


    def forward(self, x): 
        # downsample stage 
        x = self.downsample_1(x)
        x1 = self.downsample_2(x)
        # stages 1

        x1 = self.stage_1_1(x1)
        x1, x2 = self.fusion_stage_1([x1])

        # stage 2
        x1 = self.stage_1_2(x1)
        x2 = self.stage_2_1(x2)
        x1, x2, x3 = self.fusion_stage_2([x1, x2])

        # stage 3 
        x1 = self.stage_1_3(x1)
        x2 = self.stage_2_2(x2)
        x3 = self.stage_3_1(x3)

        x1, x2, x3, x4 = self.fusion_stage_3([x1, x2, x3])

        # stage 4 
        x1 = self.stage_1_4(x1)
        x2 = self.stage_2_3(x2)
        x3 = self.stage_3_2(x3)
        x4 = self.stage_4_1(x4)

        x1, x2, x3, x4 = self.fusion_stage_4([x1, x2, x3, x4])
        return x 


