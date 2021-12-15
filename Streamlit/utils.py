# Utilities
import pandas as pd
import numpy as np
import cv2
import os
import random
from tqdm.autonotebook import tqdm
import random
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
# import matplotlib
# matplotlib.use('TkAgg', force=True)
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# torch utils
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
  IMG_SIZE = 512
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

  def __init__(self, imgs, width, height, transforms=None):
      self.transforms = transforms
      self.imgs = imgs
      self.height = height
      self.width = width
      
      # sorting the images for consistency
      # To get images, the extension of the filename is checked to be jpg
      # self.imgs = [image for image in sorted(os.listdir(images_dir))]
      # self.annotate = [image for image in sorted(os.listdir(annotation_dir))]
      
      # classes: 0 index is reserved for background
      # self.classes = CFG.LABELS
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
      # img_name = self.imgs[idx]
      # file_path = os.path.join(self.images_dir, img_name)
      img = convert_from_image_to_cv2(self.imgs)
      # img = cv2.imread(self.imgs)
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
      img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
      img_res /= 255.0

      # img_res = img_res.transpose(2, 0, 1)
      # print("before transforms:", img_res.shape)
      if self.transforms:
          transforms = self.transforms(image = img_res) 
          img_res = transforms['image']
      # print("after transforms: ", img_res.shape)
      return img_res

# transforms
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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
def get_model_instance_segmentation():
    num_classes = 4
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
PATH = 'C:\\Users\\YOLO4\\OneDrive\\Documents\\Compsci_IA\\Compsci-IA\\FasterRCNN_epoch_bestPrecision.pt'
model = get_model_instance_segmentation()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(PATH, map_location=device)['model_state_dict'])
model.eval()
model.to(device)
print("Completed loading model")
def get_predictions(image, threshold=0.5):
  # PATH = 'C:/Users/YOLO/OneDrive/Desktop/github-test/Streamlit/apps/FasterRCNN_epoch_bestPrecision.pt'
  imgs = FaceMaskDataset(image, CFG.IMG_SIZE, CFG.IMG_SIZE, transforms = get_transform(False))
  # print(imgs.shape)
  imgs = imgs[0]
  imgs = imgs.to(device)
  # print(imgs.shape)
  output = model([imgs])
  output = output[0]

  return imgs, output

# postprocessing
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min x- max y-max
    # a.imshow(img)

    mask_dic = {1:'Without Mask', 2:'With Mask', 3:'Mask Weared Incorrect'}
    
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(target['boxes']):
        label = mask_dic[int(target['labels'][i].data)]
        score = int((target['scores'][i].data) * 100)

        xmin, ymin, xmax, ymax  = box[0], box[1], box[2], box[3]
        xmin = xmin.detach().numpy()
        ymin = ymin.detach().numpy()
        xmax = xmax.detach().numpy()
        ymax = ymax.detach().numpy()
        draw.rectangle(((xmin, ymin), (xmax , ymax)), outline='red')

        draw.text((xmin-20, ymin-20), f"{label} : {score}%", font=ImageFont.truetype("arial.ttf", 15), fill = 'red')

        print(f"{label} : {score}%")
        # rect = patches.Rectangle((x, y),
        #                          width, height,
        #                          linewidth = 2,
        #                          edgecolor = 'r',
        #                          facecolor = 'none')
        # Draw the bounding box on top of the image
        
        print("bounding box finished")
    return img
        # a.add_patch(rect)

def apply_nms(orig_prediction, iou_thresh=0.3):
    
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
# pick one image from the test set
# Final

