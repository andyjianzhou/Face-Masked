# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.autonotebook import tqdm
import random
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# numba
import numba
from numba import jit
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# xml library for parsing xml files
from xml.etree import ElementTree as et
import cv2

import os
import glob

class CFG:
  BATCH_SIZE = 64
  VAL_BS = 8
  TRAIN_BS = 4
  EPOCHS = 25
  IMG_SIZE = 256
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
  LABELS = [_, 'without_mask','with_mask','mask_weared_incorrect']

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(CFG.SEED)

images_dir = '../input/face-mask-detection/images/'
annotations_dir = '../input/face-mask-detection/annotations/'

class FaceMaskDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, annotation_dir, width, height, transforms=None):
        self.transforms = transforms
        self.images_dir = images_dir
        self.annotation_dir = annotation_dir
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(images_dir))]
        self.annotate = [image for image in sorted(os.listdir(annotation_dir))]
        
        # classes: 0 index is reserved for background
        self.classes = CFG.LABELS
    def __len__(self):
      return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        file_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0
        
        # annotation file, using Pascal Voc dataset format
        annot_filename = self.annotate[idx]
        annot_file_path = os.path.join(self.annotation_dir, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # cv2 image gives size as height x width
        wt = img.shape[1] #Width
        ht = img.shape[0] #height
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # bounding box, pascal voc format -> xmin, xmax, ymin, ymax
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            
            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            transforms = self.transforms(image = img_res, bboxes = target['boxes'], labels = labels) 
            img_res = transforms['image']
            target['boxes'] = torch.Tensor(transforms['bboxes'])
      
        return img_res, target


# check dataset
dataset = FaceMaskDataset(images_dir, annotations_dir, 500, 500)
print('length of dataset = ', len(dataset), '\n')

# getting the image and target for a test index.  Feel free to change the index.
img, target = dataset[35]
print('Image shape = ', img.shape, '\n','Target - ', target)

# Function to visualize bounding boxes in the image

def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the imag
        a.add_patch(rect)
    plt.show()
    
# plotting the image with bboxes. Feel free to change the index
img, target = dataset[3]
plot_img_bbox(img, target)


# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    
    if train:
        return A.Compose([
                            #A.HorizontalFlip(0.5),
                            #A.RandomBrightnessContrast(p=0.2),
                            #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))


# use our dataset and defined transformations
dataset = FaceMaskDataset(images_dir, annotations_dir, 480, 480, transforms= get_transform(train=True))
dataset_test = FaceMaskDataset(images_dir, annotations_dir, 480, 480, transforms= get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

val_data_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=10, shuffle=False, num_workers=4,
    collate_fn=collate_fn)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

'''
https://www.kaggle.com/pestipeti/competition-metric-details-script
'''

@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area

@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.
       The mean average precision at different intersection over union (IoU) thresholds.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision

iou_thresholds = [0.5]


class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
        
class EvalMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.image_precision = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, gt_boxes, pred_boxes, n=1):       
        """ pred_boxes : need to be sorted."""
        
        self.image_precision = calculate_image_precision(pred_boxes,
                                                         gt_boxes,
                                                         thresholds=iou_thresholds,
                                                         form='pascal_voc')
        self.count += n
        self.sum += self.image_precision * n
        self.avg = self.sum / self.count

def train_one_epoch(num_epochs, train_loader, model, device, optimizer):
    loader = tqdm(train_loader, total=len(train_loader))
    summary_loss = AverageMeter()
    model.train()
    i = 0    
    epoch_loss = 0
    for imgs, annotations in loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
#         print(loss_dict)
        loss = sum(loss for loss in loss_dict.values())        
        optimizer.zero_grad()
        loss.backward()
        summary_loss.update(loss.detach().item(), CFG.TRAIN_BS)
        optimizer.step() 
#         print(f'Iteration: {i}/{len(loader)}, Loss: {losses}')
#         epoch_loss += losses.item()
    print(f'Loss after epoch {num_epochs+1} = ',summary_loss.avg)
    return summary_loss.avg
    
def val_one_epoch(num_epochs, val_loader, model, device, optimizer):
    model.eval()
    epoch_loss = 0
    loader = tqdm(val_loader, total=len(val_loader))
#     summary_losses = AverageMeter()
    eval_scores = EvalMeter()
    for imgs, annotations in loader:
        imgs = list(img.to(device) for img in imgs)
        annotation = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        outputs = model(imgs)
        
        for i, image in enumerate(imgs):
            gt_boxes = annotations[i]['boxes'].data.cpu().numpy()
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].detach().cpu().numpy()
            
            preds_sorted_idx = np.argsort(scores)[::1]
            preds_sorted_boxes = boxes[preds_sorted_idx]
            eval_scores.update(pred_boxes = preds_sorted_boxes, gt_boxes = gt_boxes)
        #Uncomment for mode.train() evaluation if you want loss only
#         losses = sum(loss for loss in loss_val_dict.all())
#         summary_losses.update(summary_losses.item(), CFG.VAL_BS)
    print("Precision is: ", eval_scores.avg)
#     print('Validation loss = ', summary_losses.avg)
    return eval_scores.avg
                
#                 prediction = model([img.to(device)])[0]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
model = get_model_instance_segmentation(num_classes)

def engine():
# to train on gpu if selected.
#     num_classes = 4
    # get the model using our helper function
#     model = get_model_instance_segmentation(num_classes)
    num_epochs = CFG.EPOCHS
    # move model to the right device
    model.to(device)



    # parameters construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    precision_max = 0.
    loss_min=99999
    loss = []
    # insert train_one_epoch and val_one_epoch
    for epochs in range(num_epochs):
        print(f"============Epoch: {epochs+1}============")
        losses_train = train_one_epoch(epochs, data_loader, model, device, optimizer)
        
        precision_val = val_one_epoch(epochs, val_data_loader, model, device, optimizer)
        loss.append(precision_val)
        
        if losses_train < loss_min:
            PATH = f'./FasterRCNN_epoch_bestLosses.pt'
            torch.save({
                'epoch':epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses_train
            }, PATH)
            print(f'Best loss found in {epochs+1}, with loss of {losses_train}... saving model to {PATH}')
            loss_min = losses_train
                  
        if precision_val > precision_max:
            PATH = f'./FasterRCNN_epoch_bestPrecision.pt'
            torch.save({
                'epoch':epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses_train
            }, PATH)
            print(f'Best precision found in {epochs+1}, with precision of {precision_val}... saving model to {PATH}')
            precision_max = precision_val
        
        if losses_train < loss_min and precision_val < precision_max:
            PATH = f'./FasterRCNN_epoch_bestModel.pt'
            torch.save({
                'epoch':epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses_train
            }, PATH)
            print(f'Best overall score found in {epochs+1}, with loss of {losses_train}, and precision is {precision_val}... saving model to {PATH}')
            precision_max = precision_val
            loss_min = losses_train
        
        torch.cuda.empty_cache()
            
#Run engine
engine()