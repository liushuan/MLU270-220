from __future__ import division

from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision import models
import numpy
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# set pretrained model path
TORCH_HOME = os.getenv('TORCH_HOME')
if TORCH_HOME==None:
    print("Warning: please set environment variable TORCH_HOME such as $PWD/models/pytorch") 
    exit(1)
TORCH_HOME = TORCH_HOME+'/int8/'
os.environ['TORCH_HOME'] = TORCH_HOME

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='./img', help='path to dataset')
parser.add_argument('--class_path', type=str, default='exam.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=512, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument("--half_input", dest = 'half_input', help =
                    "the input data type, 0-float32, 1-float16/Half, default 1.",
                    default = 1, type = int)
opt = parser.parse_args()
print(opt)

cuda = False
mlu = True

if not os.path.exists('output'):
    os.makedirs('output')

# Set up model and load weights
model = models.object_detection.yolov3(pretrained=False, img_size=opt.img_size,
                            conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

model.eval().float() # Set in evaluation mode



if cuda:
    model.cuda()
elif mlu:
    mean = [0.0, 0.0, 0.0]
    std  = [1.0, 1.0, 1.0]
    print("start")
    model = models.quantization.object_detection.yolov3(pretrained=True,
                                                        quantize=True,
                                                        img_size=416,
                                                        conf_thres=0.001,
                                                        nms_thres=0.5)
    model.to(ct.mlu_device())

randn_input = torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size).float()
if opt.half_input:
    randn_input = randn_input.type(torch.HalfTensor)
model = torch.jit.trace(model, randn_input.to(ct.mlu_device()), check_trace = False)
dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(torch.HalfTensor)) if opt.half_input else Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        if mlu:
            input_imgs = input_imgs.to(ct.mlu_device())
        detections = model(input_imgs)
        if mlu:
            detections = detections.cpu().type(torch.FloatTensor) if opt.half_input else detections.cpu()
            detections = get_boxes(detections, opt.batch_size, img_size=opt.img_size)
        else:
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print ('\nSaving images:')

# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
