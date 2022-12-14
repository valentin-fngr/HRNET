

experiment_name = "are_things_working"

# paths 

summary_path = "./runs/"


INPUT_IMAGE = (480, 640)
INPUT_BBOX_SIZE = (256, 192)
HEATMAP_SIZE = (64, 48)

epochs = 250
batch_size = 16
device = "cuda"
nb_stages = 4 
nb_blocks = 4 
nb_channels = 256
bottle_neck_channels = 64
nb_joints = 17
lr = 1e-3