import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#Opencv and torch utils
import cv2
import torch
from faster_utils import FaceMaskDataset, get_predictions, get_model_instance_segmentation, torch_to_pil, plot_img_bbox, apply_nms

