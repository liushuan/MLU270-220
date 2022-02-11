from __future__ import division

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
from datetime import datetime, timedelta
import json
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
import logging
from random import sample
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

#configure logging path
logging.basicConfig(level=logging.INFO, format= '%(levelname)s: %(message)s')
logger = logging.getLogger("__name__")

torch.set_grad_enabled(False)
class Timer:
    def __init__(self):
        self.start_ = datetime.now()

    def elapsed(self):
        duration_ = datetime.now() - self.start_
        return duration_.total_seconds()

class redirect:
    content = ""
    def write(self, str):
        self.content += str
    def flush(self):
        self.content = ""

def saveResult(imageNum,batch_size,top1,top5,meanAp,hardwaretime,endToEndTime,threads=1):
    if not os.getenv('OUTPUT_JSON_FILE'):
        return
    TIME=-1
    hardwareFps=-1
    endToEndFps=-1
    latencytime=-1
    if hardwaretime!=TIME:
        hardwareFps=imageNum/hardwaretime
        latencytime = hardwaretime / (imageNum / batch_size) * 1000
    if endToEndTime!=TIME:
        endToEndFps=imageNum/endToEndTime
    if top1!=TIME:
        top1=top1/imageNum*100
    if top5!=TIME:
        top5=top5/imageNum*100
    result={
            "Output":{
                "Accuracy":{
                    "top1":'%.2f'%top1,
                    "top5":'%.2f'%top5,
                    "meanAp":meanAp
                    },
                "HostLatency(ms)":{
                    "average":'%.2f'%latencytime,
                    "throughput(fps)":"%.2f"%endToEndFps,
                    # "endToEndTime":endToEndTime,
                    # "endToEndFps":endToEndFps
                    }
                }
            }
    with open(os.getenv("OUTPUT_JSON_FILE"),"a") as outputfile:
        json.dump(result,outputfile,indent=4,sort_keys=True)
        outputfile.write('\n')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

abs_path = sys.path[0]+'/'
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--data_config_path", type=str, default="../../data/coco/coco.data", help="path to data config file")
parser.add_argument("--class_path", type=str, default=abs_path+"../../data/coco/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=False, help="whether to use cuda if available")
parser.add_argument('--mlu', default=True, type=str2bool, help='Use mlu to train model')
parser.add_argument('--jit', default=True, type=str2bool, help='Use jit for inference net')
parser.add_argument("--parallel", type=int, default=1, help="Use model parallel")
parser.add_argument('--save_pt', default=False, type=str2bool, help='Whether need to save pt file')
parser.add_argument('--offline_mode', default=False, type=str2bool, help='run on pt offline mode')
parser.add_argument('--mname', default='yolov3_offline.pt', type=str, help='Name of the .pt offline file')
parser.add_argument('--image_number',type=int,help='test image number')
parser.add_argument("--quantized_mode", dest = 'quantized_mode', help =
                    "the data type, 0-float16 1-int8 2-int16 3-c_int8 4 c_int16, default 1.",
                    default = 1, type = int)
parser.add_argument("--ann_dir", dest = "ann_dir", help ="The annotation file directory",
                    default = '',type = str)
parser.add_argument("--coco_path", dest = "coco_path", help ="The coco image file directory",
                    default = '',type = str)
parser.add_argument("--data_type", dest = "data_type", help ="The data type. e.g. val2014, val2015, val2017",
                    default = 'val2014',type = str)
parser.add_argument("--json_name", dest = "json_name", help ="name of the output file(.json)",
                    default = 'results',type = str)
parser.add_argument("--half_input", dest = 'half_input', help =
                    "the input data type, 0-float32, 1-float16/Half, default 1.",
                    default = 1, type = int)
parser.add_argument('--core_number', default=16, type=int, help='Core number of mfus and offline model with simple compilation.')
parser.add_argument("--input_channel_order", default=0, type=int,
                    help="Channel order of inputs in first conv, 0-rgba, 1-argb, 2-bgra, 3-abgr")
parser.add_argument('--mcore', default='MLU270', type=str, help="Set MLU Architecture")
parser.add_argument('--quantization', default=False, type=str2bool,
                    help='Whether to quantize yolov3, set to True will quantize yolov3 not run it')
parser.add_argument('--quantized_model_path', default='.', type=str,
                    help='Quantized model path')

opt = parser.parse_args()

assert opt.coco_path != "", "coco_path is NULL, please set the path of coco images."
assert opt.ann_dir != "", "ann_dir is NULL, please set the path of annotations."
assert opt.image_number % opt.batch_size == 0, "image_number should be divided by batch_size currently."
# set pretrained model path
TORCH_HOME = os.getenv('TORCH_HOME')
if TORCH_HOME==None:
    print("Warning: please set environment variable TORCH_HOME such as $PWD/models/pytorch")
    exit(1)
if opt.quantized_mode == 0 or opt.quantization:
    TORCH_HOME = TORCH_HOME + '/origin/'
elif opt.quantized_mode == 1:
    TORCH_HOME = TORCH_HOME + '/int8/'
elif opt.quantized_mode == 2:
    TORCH_HOME = TORCH_HOME + '/int16/'
elif opt.quantized_mode == 3:
    TORCH_HOME = TORCH_HOME + '/c_int8/'
elif opt.quantized_mode == 4:
    TORCH_HOME = TORCH_HOME + '/c_int16/'
os.environ['TORCH_HOME'] = TORCH_HOME

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = abs_path+data_config["valid"]
num_classes = int(data_config["classes"])

if opt.mlu:
  ct.set_core_number(opt.core_number)
  ct.set_core_version(opt.mcore)
  ct.set_input_format(opt.input_channel_order)

model = None
# running offline mode
if opt.offline_mode:
    opt.mlu = True
    pt_file = os.path.join(os.getcwd(), opt.mname)
    if os.path.exists(pt_file):
        model = torch.jit.load(pt_file)
        model.eval().float()
    else:
        print('No offline file, Please provide the .pt file!')
        sys.exit()
else:
    # Initiate model
    if opt.mlu:
        model = models.quantization.object_detection.yolov3(pretrained=True,
                                                            quantize=True,
                                                            img_size=416,
                                                            conf_thres=0.001,
                                                            nms_thres=0.5)
        model.to(ct.mlu_device())
    else:
        model = models.object_detection.yolov3(pretrained=True, img_size=opt.img_size, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    if opt.use_cuda:
        model = model.cuda()

    model.eval().float()

    if opt.quantization:
        mean = [0.0, 0.0, 0.0]
        std  = [1.0, 1.0, 1.0]
        dtype = "int16" if opt.quantized_mode == 2 or opt.quantized_mode == 4 else "int8"
        per_channel = False if opt.quantized_mode == 1 or opt.quantized_mode == 2 else True
        qconfig = {'iteration': opt.image_number, 'use_avg':False, 'data_scale':1.0, 'mean': mean, 'std': std, 'per_channel': per_channel}
        quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype=dtype, gen_quant=True)

    if opt.jit:
        # trace network
        example = torch.randn(opt.batch_size, 3, opt.img_size, opt.img_size).float()
        trace_input = torch.randn(1, 3, opt.img_size, opt.img_size).float()
        if opt.half_input:
            example = example.type(torch.HalfTensor)
            trace_input = trace_input.type(torch.HalfTensor)
        model = torch.jit.trace(model, trace_input.to(ct.mlu_device()), check_trace = False)
        if opt.save_pt:
            model(example.mlu())
            pt_file = os.path.join(os.getcwd(), opt.mname)
            model.save(pt_file)


# Get dataloader
if opt.mlu:
    dataset = ListDataset(opt.coco_path, test_path, normalize = False)
else:
    dataset = ListDataset(opt.coco_path, test_path, normalize = True)

if opt.image_number and opt.image_number<len(dataset):
    dataset.img_files = dataset.img_files[0:opt.image_number]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor

logger.info('Compute mAP...')

all_detections = []
all_annotations = []

seen = 0
total_e2e = 0
total_hardware = 0
jdict, stats, img_ids = [], [], []
coco91class = coco80_to_coco91_class()
for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

    timer = Timer()

    imgs = Variable(imgs.type(torch.HalfTensor)) if opt.half_input and opt.mlu else Variable(imgs.type(Tensor))

    with torch.no_grad():
        if opt.mlu:
            imgs = imgs.to(ct.mlu_device())
        if opt.quantization:
            outputs = quantized_model(imgs)
            continue
        timer1 = Timer()
        outputs = model(imgs)
        total_hardware += timer1.elapsed()

        if opt.mlu:
            outputs = outputs.cpu().type(torch.FloatTensor) if opt.half_input else outputs.cpu()
            outputs = get_boxes(outputs, opt.batch_size, img_size=opt.img_size)
        else:
            outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres,
                                          nms_thres=opt.nms_thres)
    total_e2e += timer.elapsed()

    for si, pred in enumerate(outputs):
        img = np.array(Image.open(img_path[si]))
        img_h, img_w, _ = img.shape
        img_name = os.path.splitext(os.path.basename(img_path[si]))[0]
        image_id = int(img_name.split('_')[-1].lstrip('0'))
        img_ids.append(image_id)
        box = pred[:, :4].clone()   #x1, y1, x2, y2

        scaling_factors = min(float(opt.img_size) / img_w, float(opt.img_size) / img_h)

        box[:, 0] = ((box[:, 0] - (opt.img_size - scaling_factors * img_w) / 2.0) / scaling_factors) / img_w
        box[:, 1] = ((box[:, 1] - (opt.img_size - scaling_factors * img_h) / 2.0) / scaling_factors) / img_h
        box[:, 2] = ((box[:, 2] - (opt.img_size - scaling_factors * img_w) / 2.0) / scaling_factors) / img_w
        box[:, 3] = ((box[:, 3] - (opt.img_size - scaling_factors * img_h) / 2.0) / scaling_factors) / img_h

        for di, d in enumerate(pred):
            box_temp = []
            box_temp.append(round(box[di][0].item(), 3) * img_w)
            box_temp.append(round(box[di][1].item(), 3) * img_h)
            box_temp.append((round(box[di][2].item(), 3) - round(box[di][0].item(), 3))  * img_w)
            box_temp.append((round(box[di][3].item(), 3) - round(box[di][1].item(), 3))  * img_h)
            jdict.append({'image_id': image_id,
                          'category_id': coco91class[int(d[6])],
                          'bbox': box_temp,
                          'score': round(d[5].item(), 5)})

if opt.quantization:
  checkpoint = quantized_model.state_dict()
  torch.save(checkpoint, '{}/yolov3.pth'.format(opt.quantized_model_path))

else:
  json_file_name = '%s.json'%(opt.json_name)
  print(json_file_name)
  with open(json_file_name, 'w') as file_json:
      json.dump(jdict, file_json)

  data_dir = opt.ann_dir
  data_type = opt.data_type

  ann_type = ['segm', 'bbox', 'keypoints']
  ann_type = ann_type[1]
  prefix = 'person_keypoints' if ann_type == 'keypoints' else 'instances'

  ann_file = '%s/annotations/%s_%s.json'%(data_dir, prefix, data_type)
  coco_gt = COCO(ann_file)

  coco_dt = coco_gt.loadRes(json_file_name)

  coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
  coco_eval.params.imgIds = img_ids
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  mAP = round(coco_eval.stats[1], 3)

  logger.info('Global accuracy:')
  logger.info('throughput: ' + str(len(dataset) / total_e2e))
  logger.info('latency: '+ str(opt.batch_size / (len(dataset)/total_hardware) * 1000))
  logger.info('Mean AP: ' + str(mAP))
  # logger.info('End2end throughput fps: ' + str(len(dataset) / total_e2e))
  saveResult(len(dataset), opt.batch_size, -1, -1, mAP, total_hardware, total_e2e)

# input_json_file = os.getenv('OUTPUT_JSON_FILE','')
# if os.path.isfile(input_json_file):
#     r = redirect()
#     sys.stdout = r
#     coco_eval.summarize()
#     file_in = open(input_json_file, "r")
#     json_data = json.load(file_in)
#     update_json_meanAp(json_data, r.content, 'meanAp')
#     file_in.close()
#     file_out = open(input_json_file, "w")
#     json.dump(json_data, file_out,indent=2)
#     file_out.close()
#     r.flush()
