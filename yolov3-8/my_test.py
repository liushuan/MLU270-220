import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from torch.utils.data import DataLoader
from utils.utils import *

import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

from torchvision.models.quantization.utils import quantize_model

def get_data_set(path, imgsz, batch_size, single_cls=False):

    dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=single_cls, pad=0.5)
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)
    return dataloader
def quantization_model():

    weights = r'weights/unzip.pt'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = (512, 512)
    # Initialize
    device = torch.device('cpu')
    # Initialize model
    model = Darknet(cfg, imgsz)
    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    # Eval mode
    model.to(device).eval().float()
    
    quantization = True
    quantized_mode = 1
    image_number = 1
    if quantization:
        mean = [0.0, 0.0, 0.0]
        std  = [255, 255, 255]
        dtype = "int16" if quantized_mode == 2 or quantized_mode == 4 else "int8"
        per_channel = False if quantized_mode == 1 or quantized_mode == 2 else True
        
        qconfig = {'iteration': image_number, 'use_avg':False, 'data_scale':1.0, 'mean': mean, 'std': std, 'per_channel': per_channel, 'firstconv':False}
        quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype=dtype, gen_quant=True)
        quantized_model.eval().float()
    #inference 100.
    img_path_list = '/opt/cambricon/data/data_list1.txt'
    data_loader = get_data_set(img_path_list, imgsz[0], 1)
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(data_loader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        #targets = targets.to(device)
        #nb, _, height, width = imgs.shape  # batch size, channels, height, width
        #whwh = torch.Tensor([width, height, width, height]).to(device)
        outputs = quantized_model(imgs)[0]
        #print("output:",outputs.shape)
    
    if quantization:
        checkpoint = quantized_model.state_dict()
        torch.save(checkpoint, 'weights/best_qua.pth')
    
    #mlu = True
    #if mlu:
    #    ct.set_core_number(4)
    #    ct.set_core_version("MLU279")
    #    ct.set_input_format(0)
    
    '''   
    jit = True
    batch_size = 1
    save_pt = True
    half_input = False
    mname = "./weights/best.jit"
    if jit:
        #model = model.to(ct.mlu_device())
        # trace network
        example = torch.randn(batch_size, 3, imgsz[0], imgsz[1]).float()
        trace_input = torch.randn(1, 3, imgsz[0], imgsz[1]).float()
        if half_input:
            example = example.type(torch.HalfTensor)
            trace_input = trace_input.type(torch.HalfTensor)
        model = torch.jit.trace(model, trace_input.to(device), check_trace = False)
        
        if save_pt:
            #model(example.mlu())
            pt_file = mname
            model.save(pt_file)
    
    '''

def val_model():
    qua_weights = r'weights/best_qua.pth'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = (512, 512)
    # Initialize
    device_cpu = torch.device('cpu')
    device = ct.mlu_device()
    # Initialize model
    quantized_model = Darknet(cfg, img_size=512, conf_thres=0.2, nms_thres=0.5).eval()
    #quantize_model
    #quantized_model = quantize_model(quantized_model)
    quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(quantized_model)
    mlu = True
    if mlu:
        # Load weights
        quantized_model.load_state_dict(torch.load(qua_weights, map_location=device_cpu), strict=False)
        # Eval mode
        quantized_model = quantized_model.to(device)
        quantized_model.eval().float()
    
   
    '''
    jit = True
    batch_size = 1
    save_pt = True
    half_input = False
    mname = "./weights/best.jit"
    
    if jit:
        # trace network
        example = torch.randn(batch_size, 3, imgsz[0], imgsz[1]).float()
        trace_input = torch.randn(1, 3, imgsz[0], imgsz[1]).float()
        
        quantized_model = torch.jit.trace(quantized_model, trace_input.to(device_cpu), check_trace = False)
        
        if save_pt:
            pt_file = mname
            quantized_model.save(pt_file)
    print("jit save finished")
    '''
    
    '''
    #inference 100.
    img_path_list = '/opt/cambricon/data/data_list1.txt'
    data_loader = get_data_set(img_path_list, imgsz[0], 1)
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(data_loader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        
        print("device type:",imgs.device.type)
        print("shape:", imgs.shape)
        #targets = targets.to(device)
        #nb, _, height, width = imgs.shape  # batch size, channels, height, width
        #whwh = torch.Tensor([width, height, width, height]).to(device)
        outputs = quantized_model(imgs)[0]
        #print("output:",outputs.shape)
    
    '''
    
    print("start jit")
    jit = True
    batch_size = 1
    save_pt = True
    half_input = False
    mname = "./weights/best.jit"
    
    if jit:
        # trace network
        #example = torch.randn(batch_size, 3, imgsz[0], imgsz[1]).float()
        ct.set_core_number(4)
        ct.set_core_version('MLU270')
        trace_input = torch.randn(1, 3, imgsz[0], imgsz[1]).float()
        quantized_model_traced = torch.jit.trace(quantized_model, trace_input.to(device), check_trace = False)
        
        quantized_model_traced(trace_input.to(device))
        
        #if save_pt:
        #    pt_file = mname
        #    quantized_model_traced.save(pt_file)
    print("jit save finished")
    
    
    
    
    
    
    
    
    
    
    

def generate_onfine():
    qua_weights = r'weights/best_qua.pth'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = (512, 512)
    # Initialize
    device_cpu = torch.device('cpu')
    device = ct.mlu_device()
    # Initialize model
    quantized_model = Darknet(cfg, img_size=512, conf_thres=0.2, nms_thres=0.5).eval()
    #quantize_model
    #quantized_model = quantize_model(quantized_model)
    quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(quantized_model)
    mlu = True
    if mlu:

        # Load weights
        quantized_model.load_state_dict(torch.load(qua_weights, map_location=device_cpu), strict=False)
        # Eval mode
        quantized_model = quantized_model.to(device)
        quantized_model.eval().float()
    
   
    '''
    jit = True
    batch_size = 1
    save_pt = True
    half_input = False
    mname = "./weights/best.jit"
    
    if jit:
        # trace network
        example = torch.randn(batch_size, 3, imgsz[0], imgsz[1]).float()
        trace_input = torch.randn(1, 3, imgsz[0], imgsz[1]).float()
        
        quantized_model = torch.jit.trace(quantized_model, trace_input.to(device_cpu), check_trace = False)
        
        if save_pt:
            pt_file = mname
            quantized_model.save(pt_file)
    print("jit save finished")
    '''
   
    jit = True
    print("### jit")
    if jit:
            batch_size = 1
            #torch.set_grad_enabled(False)
            ct.set_core_number(4)
            ct.set_core_version('MLU220')
            ct.save_as_cambricon('yolov3_int8_4_'+str(batch_size))
            
            trace_input = torch.randn(1, 3, imgsz[0], imgsz[1], dtype=torch.float)
            input_mlu_data = trace_input.to(ct.mlu_device())
            quantized_model_traced = torch.jit.trace(quantized_model, input_mlu_data, check_trace = False)
            
            
            example = torch.randn(batch_size, 3, imgsz[0], imgsz[1]).float()
            
            quantized_model_traced(example.to(device))
    
    print("end jit")
    #inference 100.
    '''
    img_path_list = '/opt/cambricon/data/data_list1.txt'
    data_loader = get_data_set(img_path_list, imgsz[0], 1)
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(data_loader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        
        print("device type:",imgs.device.type)
        print("shape:", imgs.shape)
        #targets = targets.to(device)
        #nb, _, height, width = imgs.shape  # batch size, channels, height, width
        #whwh = torch.Tensor([width, height, width, height]).to(device)
        outputs = quantized_model(imgs)
        #print("output:",outputs.shape)
    '''

if __name__ == '__main__':
    with torch.no_grad():
        #quantization_model()
        #val_model()
        generate_onfine()