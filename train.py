import torch 
import numpy as np 
from model.hrnet_v2 import HeatMapRegressionLoss, HRNETV2
from data.coco import COCODataset
from torch.utils.data import DataLoader 
import config
import tqdm





def get_criterion(): 
    """
    Return the heatmap regression criterion
    """
    criterion = HeatMapRegressionLoss()
    criterion.to(config.device)
    return criterion 


def get_train_val_loaders(): 
    """
    Return the training and validation loaders
    """

    train_dataset = COCODataset(data_set="train2017")
    val_dataset = COCODataset(data_set="val2017")

    train_loader = DataLoader(
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=-1
    )

    val_loader = DataLoader(
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=-1
    )
    
    return train_loader, val_loader


def get_model(): 
    """
    Initialize the model
    """
    model = HRNETV2(config.nb_stages, config.nb_blocks, config.nb_channels, config.bottle_neck_channels, config.nb_joints)
    model.to(config.device)



def get_optimizer(model): 
    """
    Initialize the optimizer and the scheduler
    """
    # TODO : add scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  
    def lambda_lr(lr): 
        if lr == 170: 
            return 1e-4
        elif lr == 200: 
            return 1e-5
        return lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, config.lr, lambda_lr)

    return optimizer, scheduler




def train(train_loader, model, optimizer, scheduler, writer, epoch): 

    for idx, batch in enumerate(tqdm(train_loader)): 
        
        image_array, keypoints_heatmap, _ = batch
        # move to device 

        image_array.to(config.device)
        keypoints_heatmap.to(config.device)

        preds = model(image_array)
        