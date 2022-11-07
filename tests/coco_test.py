import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import torch 
from torch.utils.data import DataLoader
from data.coco import COCODataset
import unittest
import os 

import torchvision.transforms as transforms


class TestCOCODataset(unittest.TestCase):


    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        coco = COCODataset()
        self.coco = coco
        self.loader = DataLoader(coco, batch_size=1)


    def test_can_generate_heatmaps(self): 

        # iterate trough the a few images and save the heatmaps 

        for i, (img, heatmap) in enumerate(self.loader): 
            assert heatmap.shape[1] == self.coco.num_joints
            
            if i >= 2: 
                break
            img = img.detach().numpy()[...,::-1]      
            h, w = img.shape[1:3]      

            for j in range(self.coco.num_joints): 
                hmap = heatmap[0, j] 
                plt.imshow(hmap.numpy())
                plt.imshow(cv2.resize(img[0], (int(w/4), int(h/4))), alpha=0.25)
                plt.savefig(os.path.join(os.getcwd(), "assets" ,str(i) + "_" + "heatmap_" + str(j) + ".png"))
                plt.show()

            


            
            



if __name__ == "__main__": 
    unittest.main()
