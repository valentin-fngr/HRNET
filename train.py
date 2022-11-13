import torch 
import numpy as np 
from model.hrnet_v2 import HeatMapRegressionLoss, HRNETV2
from data.coco import COCODataset
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms
import config
import hydra
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from omegaconf import OmegaConf


def get_criterion(cfg): 
    """
    Return the heatmap regression criterion
    """
    criterion = HeatMapRegressionLoss()
    criterion.to(cfg.config.training.device)
    return criterion 


def get_train_val_loaders(cfg): 
    """
    Return the training and validation loaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = COCODataset(data_set="train2017", transform=transform)
    val_dataset = COCODataset(data_set="val2017", transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.config.training.batch_size, 
        shuffle=True, 
        num_workers=3
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.config.training.batch_size, 
        shuffle=True, 
        num_workers=3
    )
    
    return train_loader, val_loader


def get_model(cfg): 
    """
    Initialize the model
    """
    model = HRNETV2(cfg.config.model.nb_stages, cfg.config.model.nb_blocks, cfg.config.model.nb_channels, cfg.config.model.bottle_neck_channels, cfg.config.dataset.nb_joints)
    model.to(cfg.config.training.device)
    return model 



def get_optimizer(model,cfg): 
    """
    Initialize the optimizer and the scheduler
    """
    # TODO : add scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.config.training.learning_rate)  
    def lambda_lr(lr): 
        if lr == 170: 
            return 1e-4
        elif lr == 200: 
            return 1e-5
        return lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    return optimizer, scheduler



def train(train_loader, model, criterion, optimizer, scheduler, writer, epoch, cfg): 
    """
    Train the model 
    
    
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader class
        Training dataloader
    
    model : torch model 
        the hrnetv2 model 
    criterion: 
        The hrnetv2 loss, i.e heatmap regression loss
    optimizer: 
        optimizer instance 
    scheduler :
        learnin rate scheduler instance 
    writer:
        The tensorboard writer 
    epoch: int 
        Current epoch 

    """

    loss = 0.0
    print(f"--- Training at epoch : {epoch} ---")
    for idx, batch in enumerate(tqdm(train_loader)):  
        
        image_array, keypoints_heatmap, _ = batch
        batch_size = image_array.shape[0]
        # move to device 
        image_array = image_array.to(cfg.config.training.device)
        keypoints_heatmap = keypoints_heatmap.to(cfg.config.training.device)        
        preds = model(image_array)
        # compute loss 
        
        current_loss = criterion(preds, keypoints_heatmap)
        loss += current_loss.item()
        
        if idx % 60 == 0: 
            # monitor metric 
            writer.add_scalar("Train/Loss", loss/20.0, epoch*len(train_loader) + idx + 1)
            print(f"Epoch={epoch} - Batch size={batch_size} | Training loss = {loss / 20.0}")
            loss = 0.0 

        # backpropagate 
        current_loss.backward()
        optimizer.step()

        model.zero_grad()


    scheduler.step()
    print("Next learning rate : ", scheduler.get_last_lr())
    writer.add_scalar("Lr_Scheduler", scheduler.get_last_lr(), epoch)

    print(f"--- Training at epoch end: {epoch} ---","\n")



def visualize_preds(data_loader, model, writer, epoch, is_train=True): 
    """
    Visualize the model's predictions using tensorboard 

    Parameters
    ----------

    data_loader : torch.utils.data.DataLoader
        the data to use for running predictions
    model: pytorch model 
        The model to use for predicting the keypoints 
    writer: 
        Tensorboard writer to monitor and visualize the predictions
    epoch: int
        The current epoch 
    is_train: 
        True if the training data are being used for predictions
    """
    

    with torch.no_grad(): 

        # get data 

        for i, batch in enumerate(tqdm(data_loader)): 
            
            image_array, keypoints_heatmap, _ = batch
            # compute preds 
            preds = model(image_array)
            # visualize preds and ground_truth 

            nb_keypoints = keypoints_heatmap[1]
            for j in range(nb_keypoints): 

                # visualize gt 
                writer.add_image("gt_image", keypoints_heatmap[i][j].cpu().numpy(), epoch)
                # visualize preds
                writer.add_image("pred_image", preds[i][j].cpu().numpy(), epoch)

            if i > 2: 
                break 


@hydra.main(config_name='config/config.yaml')
def main(cfg): 
    """
    Main function
    """
    print("--- Loading configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("--- loading configuration : DONE ---", "\n")

    print("--- Loading training data ---")
    train_loader, val_loader = get_train_val_loaders(cfg)
    print(" Loading training data : DONE ---", "\n")

    print("--- Loading criterion ---")
    criterion = get_criterion(cfg)
    print("--- Loading criterion : DONE ---", "\n")

    print("--- loading model ---")
    model = get_model(cfg)
    print("--- loading model : DONE ---", "\n")
    print("--- Loading optimizer and scheduler ---")
    optimizer, scheduler = get_optimizer(model, cfg)
    print("--- Loading optimizer and scheduler ---")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(config.summary_path + "are_things_working_", current_time)

    # main loop 

    print("--- Training : starting ---")
    for epoch in range(config.epochs): 
        
        # train 
        train(train_loader, model, criterion, optimizer, scheduler, writer, epoch, cfg)
        # validation
         

        


if __name__ == "__main__": 

    main()