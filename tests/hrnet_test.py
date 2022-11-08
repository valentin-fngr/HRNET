import unittest 
import numpy as np 
import torch 
from model.hrnet_v2 import FusionModule, HRNETV2, Hrnet_v2_head





class TestHRNETModules(unittest.TestCase):


    def test_forward_fusion_module_add_branch(self): 
        """
        Test the successful forward pass of the fusion module  
        """
        nb_input = 4
        nb_channels = 100
        size = 256
        
        x = [torch.randn(1, nb_channels*(k+1), int(size/(2**k)), int(size/(2**k))) for k in range(nb_input)]
        module = FusionModule(nb_input, nb_channels)
        output = module(x)

        assert len(output) == len(x) + 1


    def test_forward_fusion_module_no_branch(self): 
        """
        Test the successful forward pass of the fusion module without adding an additional branch to the output
        """
        nb_input = 4
        nb_channels = 100
        size = 256
        add_branch = False
        
        x = [torch.randn(1, nb_channels*(k+1), int(size/(2**k)), int(size/(2**k))) for k in range(nb_input)]
        module = FusionModule(nb_input, nb_channels, add_branch)
        output = module(x)

        for item in output: 
            assert item.shape[0] == 1        
        assert len(output) == len(x)


    def test_can_instantiate_hrnet_v2(self): 
        """
        Test the successful instantiation of hrnet_v2
        """ 
        nb_stages = 4 
        nb_blocks = 4 
        nb_channels = 256
        bottleneck_channels = 64
        c_out = 17
        head = Hrnet_v2_head
        model = HRNETV2(nb_stages, nb_blocks, nb_channels, bottleneck_channels, c_out=c_out, head=head)

        self.assertTrue(isinstance(model, HRNETV2))

        print("Successfully instantiated HRNETV2")
        print(model)

    def test_forwad_pass_hrnet_v2(self): 
        """
        Test the successful forward pass of hrnet_v2
        """
        nb_stages = 4 
        nb_blocks = 4 
        nb_channels = 100
        bottleneck_channels = 64
        c_out = 17

        head = Hrnet_v2_head
        x = torch.randn(1, 3, 256, 256)

        model = HRNETV2(nb_stages, nb_blocks, nb_channels, bottleneck_channels, c_out=c_out, head=head)
        output = model(x)
        assert output.shape == (1, c_out, 256/nb_stages, 256/nb_stages)



if __name__ == "__main__": 
    unittest.main()


