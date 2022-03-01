# Utilities
import pandas as pd
import numpy as np
import cv2
import os
import random
from tqdm.autonotebook import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
# =============================================================================
import torch
import torchvision
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# from apps import FasterRCNN_epoch_bestPrecision.pt, FasterRCNN_epoch_bestLoss.pt
#Albumentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#Streamlit
import streamlit as st
#Hyperparameters
class CFG:
  BATCH_SIZE = 64
  VAL_BS = 8
  TRAIN_BS = 4
  EPOCHS = 25
  IMG_SIZE = 650
  NUM_WORKERS = 8
  SEED = 42069
  LR = 1e-4
  MIN_LR = 1e-6 # CosineAnnealingWarmRestarts
  WEIGHT_DECAY = 1e-6
  MOMENTUM = 0.9
  T_0 = EPOCHS # CosineAnnealingWarmRestarts
  MAX_NORM = 1000
  T_MAX = 5
  ITERS_TO_ACCUMULATE = 1
#   BASE_OPTIMIZER = SGD #for Ranger
  OPTIMIZER = 'Adam' # Ranger, AdamW, AdamP, SGD
  MEAN = [0.485, 0.456, 0.406]
  STD = [0.229, 0.224, 0.225]
  N_FOLDS = 5
  START_FOLDS = 0
  # LABELS = [_, 'without_mask','with_mask','mask_weared_incorrect']

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# preprocessing
# defining the files directory and testing directory
# images_dir = '../input/face-mask-detection/images/'
# annotations_dir = '../input/face-mask-detection/annotations/'
class FaceMaskDataset(torch.utils.data.Dataset):

  def __init__(self, imgs, width, height, transforms=None, real_time = True):
      self.transforms = transforms
      self.imgs = imgs
      self.height = height
      self.width = width
      self.real_time = real_time
      
      # sorting the images for consistency
      # To get images, the extension of the filename is checked to be jpg
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
      # img_name = self.imgs[idx]
      # file_path = os.path.join(self.images_dir, img_name)
      if self.real_time is False:
         img = convert_from_image_to_cv2(self.imgs)
         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
      else:
          img_rgb = cv2.cvtColor(self.imgs, cv2.COLOR_BGR2RGB).astype(np.float32)
      # img = cv2.imread(self.imgs)
      img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
      img_res /= 255.0
      if self.transforms:
          transforms = self.transforms(image = img_res) 
          img_res = transforms['image']
      # print("after transforms: ", img_res.shape)
      return img_res

# transforms
    # return np.asarray(img)

def get_transform(train):
    
    if train:
        return A.Compose([
                            #A.HorizontalFlip(0.5),
                            #A.RandomBrightnessContrast(p=0.2),
                            #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ],)
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ],)
  ## model + reqs
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
#predictions
def collate_fn(batch):
    return tuple(zip(*batch))


print("Completed loading model")
def get_predictions(image, width, height, PATH, real_time = False):
    if PATH == 'UploaderDet':
        PATH = 'C:\\Users\\YOLO4\\OneDrive\\Documents\\Compsci_IA\\Compsci-IA\\UploaderDet\\FasterRCNN_epoch_bestPrecision.pt'
        num_classes = 4
    elif PATH == 'FriendDet':
        PATH = 'C:\\Users\\YOLO4\\OneDrive\\Documents\\Compsci_IA\\Compsci-IA\\FriendDet\\FasterRCNN_epoch_bestPrecision.pt'
        num_classes = 5
    
    model = get_model_instance_segmentation(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(PATH, map_location=device)['model_state_dict'])

    model.eval()
    model.to(device)
    # PATH = 'C:/Users/YOLO/OneDrive/Desktop/github-test/Streamlit/apps/FasterRCNN_epoch_bestPrecision.pt'
    imgs = FaceMaskDataset(image, width=width, height=height, transforms = get_transform(False), real_time = real_time)
    imgs = imgs[0]
    imgs = imgs.to(device) #set model to cpu, as we are not using GPU to inference/predict
    output = model([imgs])
    output = output[0] 

    return imgs, output

# postprocessing
def plot_img_bbox(img, target, PATH):

    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min x- max y-max
    if PATH == 'UploaderDet':
        mask_dic = {1:'Without Mask', 2:'With Mask', 3:'Mask Weared Incorrect'} #Labels
    elif PATH == 'FriendDet':
        mask_dic = {1:'Jun Mask Off', 2:'Nic Mask On', 3:'Nic Mask Off', 4:'Jun Mask On'} #Labels
    
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(target['boxes']):
        label = mask_dic[int(target['labels'][i].data)]
        score = int((target['scores'][i].data) * 100)
        # Plotting bounding boxes require format   xmin,xin | ymax,ymax
        xmin, ymin, xmax, ymax  = box[0], box[1], box[2], box[3] #Boxes are predictions from model
        xmin = xmin.detach().numpy() #must detach these because we are using CPU, not GPU
        ymin = ymin.detach().numpy()
        xmax = xmax.detach().numpy()
        ymax = ymax.detach().numpy()
        #draw boxes
        draw.rectangle(((xmin, ymin), (xmax , ymax)), outline='red', width = 2)
        #Draw accuracy and confidence score 
        draw.text((xmin-20, ymin-20), f"{label} : {score}%", font=ImageFont.truetype("arial.ttf", 15), fill = 'red')
    return img, label

def apply_nms(orig_prediction, iou_thresh):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    print("Converting to PIL image")
    return transforms.ToPILImage()(img).convert('RGB')
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

