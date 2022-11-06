import numpy as np 
import matplotlib.pyplot as plt
from data.coco import COCODataset 
import logging
import cv2
import os
import matplotlib.patches as patches
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)





def visualize_train_coco(nb_images=5): 
    """
        A function that visualize the bounding boxes and keypoints  
        based on the COCODataset class data
    """
    
    coco = COCODataset()
    logger.info(f"Successfully loaded {len(coco)} instances")
    
    for idx, annot in enumerate(coco.db[:nb_images]): 
        image_path = annot["image_path"]
        image_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_array = image_array[...,::-1]
        if image_array is None: 
            print("Could not load image ...")
            continue

        x,y,w,h = annot["bbox"]
        keypoints = annot["joints_2d"]
        keypoints_visibility = annot["joints_2d_visibility"]

        fig, ax = plt.subplots()
        rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # plot bbox
        ax.add_patch(rect)
        # plot keypoints
        plt.scatter(x=keypoints[:, 0], y=keypoints[:, 1], c="red")
        # save plot 
        plt.imshow(image_array)
        plt.savefig(os.path.join(os.getcwd(), "assets" ,str(idx) + image_path.split("/")[-1]))
        plt.show()
        




        










if __name__ == "__main__": 
    logger.info("--- Visualizing coco data ---")
    visualize_train_coco()