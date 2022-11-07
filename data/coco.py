from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
from pycocotools.coco import COCO
import os 
import numpy as np 
import cv2
import torch 




class COCODataset(Dataset): 



    def __init__(self, transform=None): 
        super().__init__()

        # TODO : 
        # Use config file to set all the values
        self.is_train = True
        # COCO dataset root : 
        self.root = "/home/fontanger/research/dataset/COCO/"
        # either train2017 or val2017
        self.data_set = "val2017"
        # image width and height
        self.image_width = 640
        self.image_height = 480
        self.aspect_ratio = self.image_width * 1.0 / self.image_height

        self.heatmap_width = self.image_width / 4 
        self.heatmap_height = self.image_height / 4 

        # used for gaussian filtering 
        self.sigma = 3

        # no idea what it does
        self.pixel_std = 200
        self.num_joints = 17

        self.transform = transform

        self.coco = COCO(self._get_annotation_file())
        self.image_ids = self._get_img_ids()
        print(f"Total number of images for this set : {len(self.image_ids)}")
        self.num_imgs = len(self.image_ids)
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ["__background__"] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict([(cls, i) for i, cls in enumerate(self.classes)])
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))

        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

                
        self.db = self._get_db()



    def __len__(self):
        return len(self.db)

    def _get_db(self): 
        """
            Load all instances based on data_set
        """
        if self.is_train: 
            db = self._load_coco_keypoints()

        return db



    def _generate_heatmaps(self, keypoints): 
        """
            Generate #keypoints target heatmap using 2d gaussian filtering 

            Args: 
                keypoints : nd array of size (#keypoints, 2)
            Output: 
                heatmaps : nd array of size (#keypoints, heatmap_width, heatmap_height)
        """

        kpts = keypoints.copy()
        heatmaps = np.zeros((self.num_joints, int(self.heatmap_height), int(self.heatmap_width)))
      
        heatmap_stride_h = self.image_height / self.heatmap_height 
        heatmap_stride_w = self.image_width / self.heatmap_width

        size = self.sigma * 2 + 1  # 5
        x = np.arange(0, size, 1, dtype=np.float32)
        y = np.arange(0, size, 1, dtype=np.float32)[:, None] 

        # for each keypoint : 
        for i in range(self.num_joints): 
            # rescale keypoints to heatmap size 
            x0 = y0 = size // 2
            # load gaussian kernel (+1 to unsure odd size kernel)
            g = np.exp(-((x - x0)**2 + (y - y0)**2)) / (2*self.sigma**2)

            # pixel coordinate where we will center our 2D gaussian kernel 
            mu_x = kpts[i][1] / heatmap_stride_w
            mu_y = kpts[i][0] / heatmap_stride_h

            print("old : ", kpts[i])
            print("new : ", mu_x, mu_y)

            # get upper left and bottom right (x,y) coordinates for the 2D gaussian filtering
            ul = (mu_x - self.sigma, mu_y - self.sigma)
            br = (mu_x + self.sigma + 1, mu_y + self.sigma + 1)
            # get part of the kernel that we will use for computing the heatmap
            g_x = int(max(0, -ul[0])), int(min(br[0], self.heatmap_width) - ul[0])
            g_y = int(max(0, -ul[1])), int(min(br[1], self.heatmap_height) - ul[1])
            # get upper left and bottom right pixel to modify in the heatmap 
            hmap_x = int(max(0, ul[0])), int(min(br[0], self.heatmap_width))
            hmap_y = int(max(0, ul[1])), int(min(br[1], self.heatmap_height))


            # modify heatmap 
            heatmaps[i, hmap_x[0]:hmap_x[1], hmap_y[0]: hmap_y[1]] = g[g_x[0]:g_x[1], g_y[0]: g_y[1]]

        return heatmaps


    def __getitem__(self, index):
        """
            Return a training sample

            Args: 
                index : item's index 

            Output: 
                image : 3D image array (H,W,3)
                keypoints_heatmap : 2D gaussian heatmap for each keypoint (nb_keypoints, h, w)
        """
        
        item = self.db[index]
        image_path = item["image_path"]
        keypoints = item["joints_2d"]
        keypoints_visibility = item["joints_2d_visibility"]

        image_array = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image_array is None: 
            raise ValueError(f"Image path : {image_path} returns None array")

        # rescale keypoints to heatmap size
        keypoints_heatmap = self._generate_heatmaps(keypoints)
        return image_array, keypoints_heatmap


    def _load_coco_keypoints(self): 
        """
            Load all keyppoints
        """
        keypoints = []
        for idx in self.image_ids: 
            # multiple poses per person in a single image
            keypoints.extend(self._load_single_annotation(idx))
        
        return keypoints


    def _get_img_ids(self): 
        """
            Return the list of image ids based on data_set
        """
        return self.coco.getImgIds()


    def _get_annotation_file(self):
        """
            Return the annotation file path based on data_set
        """
        # load from train annotation
        if self.is_train:
            folder_path = os.path.join(self.root, "annotations", "train")
            annotation_path = os.path.join(folder_path, "person_keypoints_" + self.data_set +".json")
    

        return annotation_path


    def _get_box_cs(self, box): 
        """
            Return the center and the scale of a bounding box of size (4,)
            
            Args: 
                box : (x,y,w,h)
            
            Output: 
                center, scale
        """

        x, y, w, h = box
        center = [None, None] 
        center[0] = x + (w - 1)/2
        center[1] = y + (h - 1)/2 

        # if the width of the bbox is higher than the image ratio, 
        # we need to scale the height accordingly 
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        # if the width of the bbox is lower than the image ration 
        # we need to scale the width accordingly
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = [h * self.pixel_std, w * self.pixel_std]
        return np.array(center, dtype=np.float32), np.array(scale, dtype=np.float32)


    def _load_single_annotation(self, index): 
        """
            Load annotations for a single image index

            Args: 
                index : item's index
            
            Ouput: 
                annotation_gt : a list of dict{"image_path", "scale", "joints_2d", "joints_2d_visibility", "bbox"}
        """

        image_obj = self.coco.loadImgs(index)[0]
        file_name = image_path = os.path.join(self.root, "images", self.data_set, image_obj["file_name"])
        image_width = image_obj["width"]
        image_height = image_obj["height"]

        annot_id = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        annot_obj = self.coco.loadAnns(annot_id)

        objs = []   
        
        annot_gt = []
        
        for obj in annot_obj:
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = x1 + max(0, w-1)
            y2 = y1 + max(0, h - 1)

            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            else: 
                continue 
            
            cls = obj["category_id"]
            # if not human 
            if cls != 1: 
                continue 
            
            # sanity check on keypoints
            if max(obj["keypoints"]) == 0: 
                continue 

            joints_2d = np.zeros((self.num_joints, 2), dtype=np.float32)
            joints_2d_visibility = np.zeros((self.num_joints, 2), dtype=np.float32)

            # retrieve keypoint 
            for joint_idx in range(self.num_joints): 
                joints_2d[joint_idx, 0] = obj["keypoints"][joint_idx * 3 + 0]
                joints_2d[joint_idx, 1] = obj["keypoints"][joint_idx * 3 + 1]
                is_visible = obj["keypoints"][joint_idx * 3 + 2]

                # if occulted or visible, set it to visible anyways
                if is_visible >= 1: 
                    is_visible = 1 

                joints_2d_visibility[joint_idx, 0] = is_visible
                joints_2d_visibility[joint_idx, 1] = is_visible
                
            center, scale = self._get_box_cs(obj['clean_bbox'])
            annot_gt.append({
                "image_path" : file_name, # get image path
                "center" : center, 
                "scale" : scale, 
                "joints_2d" : joints_2d, 
                "joints_2d_visibility": joints_2d_visibility, 
                "bbox" : obj["clean_bbox"]
            })

        return annot_gt
            




if __name__ == "__main__":

    dataset = COCODataset()
    print(len(dataset))
    

    