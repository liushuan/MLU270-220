import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from torch.utils.data import DataLoader
from utils.utils import *

import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

from torchvision.models.quantization.utils import quantize_model

image_size = (512, 512)
number_class = 29


def transform(img):
    img = img.astype('float32')
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    return img

def get_image_tensor(img_path, size_shape):
    Image = cv2.imread(img_path)
    height, width, _ = Image.shape
    if height != size_shape[0] or width != size_shape[1]:
        Image = cv2.resize(Image, size_shape)
    Image = transform(Image)
    data = Image[np.newaxis, :]
    return data  


        
def quantization_model():

    weights = r'weights/unzip.pt'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = image_size
    # Initialize
    device = torch.device('cpu')
    # Initialize model
    model = Darknet(cfg, imgsz[0],conf_thres=0.2, nms_thres=0.5)
    # Load weights
    
    for name, param in model.named_parameters():
        print(name, param.shape)
    
    model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    # Eval mode
    model.to(device).eval().float()
    
    iteration_number = 200
    qconfig = {'iteration': iteration_number, 'firstconv':False}
    quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant=True)
    quantized_model.eval().float()
    #inference 100.
    for i in tqdm(range(iteration_number)):
        input_img1 = get_image_tensor("quan_img/" + str(i)+".jpg", imgsz)
        input_img1 = torch.from_numpy(input_img1)
        outputs = quantized_model(input_img1)

    checkpoint = quantized_model.state_dict()
    torch.save(checkpoint, 'weights/best_qua.pth')
    
    input_img2 = get_image_tensor("quan_img/2.jpg", imgsz)
    input_img2 = torch.from_numpy(input_img2)
    outputs2 = quantized_model(input_img2)

    show_result("quan_img/2.jpg", outputs2, imgsz)
    
def show_result(img_path, outputs2, size_shape):
    Image = cv2.imread(img_path)
    height, width, _ = Image.shape
    if height != size_shape[0] or width != size_shape[1]:
        Image = cv2.resize(Image, size_shape)
    
    print("outputs2:", outputs2.shape)
    pred = non_max_suppression(outputs2, number_class, 0.1, 0.5)
    print(pred)
    print("len is:", len(pred)) 
    for p in pred:
        p = p.numpy()
        for i in range(p.shape[0]):
            #print (p[i][0], p[i][1], p[i][2], p[i][3])
            cv2.rectangle(Image, (p[i][0], p[i][1]), (p[i][2], p[i][3]), (0,0,255))
    cv2.imwrite("output.jpg", Image)
     
def val_model():
    qua_weights = r'weights/best_qua.pth'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = (512, 512)
    # Initialize
    device_cpu = torch.device('cpu')
    device = ct.mlu_device()
    # Initialize model
    model = Darknet(cfg, imgsz[0], conf_thres=0.2, nms_thres=0.5)
  
    quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
    quantized_model.load_state_dict(torch.load(qua_weights))
    
    # Eval mode
    #quantized_model = quantized_model.to(device)
    quantized_model.eval().float()
    
    input_img2 = get_image_tensor("quan_img/2.jpg", imgsz)
    input_img2 = torch.from_numpy(input_img2)
    outputs2 = quantized_model(input_img2)

    show_result("quan_img/2.jpg", outputs2, imgsz)
    

def show_result_mlu(img_path, outputs2, size_shape):
    Image = cv2.imread(img_path)
    height, width, _ = Image.shape
    if height != size_shape[0] or width != size_shape[1]:
        Image = cv2.resize(Image, size_shape)
    
    outputs2 = outputs2.reshape((outputs2.shape[0], outputs2.shape[1]))
    numBoxFinal = int(outputs2[0][0])
    print("num:", numBoxFinal)
    for i in range(numBoxFinal):
        c1 = (int(outputs2[0][64+i*7+3]*size_shape[0]), int(outputs2[0][64+i*7+4]*size_shape[0]))
        c2 = (int(outputs2[0][64+i*7+5]*size_shape[0]), int(outputs2[0][64+i*7+6]*size_shape[0]))
        batch_size = outputs2[0][64+i*7+0]
        class_indx = outputs2[0][64+i*7+1]
        score1 = outputs2[0][64+i*7+2]
        print("value:", batch_size, class_indx, score1, c1, c2)
        cv2.rectangle(Image, c1, c2, (0,0,255))
        #cv2.circle(Image, c1, 2, (255,0,0), -1)
        #cv2.circle(Image, c2, 2, (255,0,0), -1)
    cv2.imwrite("output.jpg", Image)
def val_model_mlu():
    qua_weights = r'weights/best_qua.pth'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = (512, 512)
    # Initialize
    device_cpu = torch.device('cpu')
    device = ct.mlu_device()
    # Initialize model
    model = Darknet(cfg, imgsz[0], conf_thres=0.2, nms_thres=0.5)
  
    quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
    quantized_model.load_state_dict(torch.load(qua_weights))
    
    # Eval mode
    quantized_model = quantized_model.to(device)
    quantized_model.eval().float()
    
    input_img2 = get_image_tensor("quan_img/2.jpg", imgsz)
    input_img2 = torch.from_numpy(input_img2)
    outputs2 = quantized_model(input_img2)
    outputs2 = outputs2.cpu().numpy()
    show_result_mlu("quan_img/2.jpg", outputs2, imgsz)    


def generate_onfine():
    qua_weights = r'weights/best_qua.pth'
    cfg = r'weights/yolov3_29cls_opt5_exam.cfg'
    imgsz = (512, 512)
    # Initialize
    device_cpu = torch.device('cpu')
    device = ct.mlu_device()
    # Initialize model
    model = Darknet(cfg, imgsz[0], conf_thres=0.2, nms_thres=0.5)
  
    quantized_model = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)
    quantized_model.load_state_dict(torch.load(qua_weights))
    
    # Eval mode
    quantized_model = quantized_model.to(device)
    quantized_model.eval().float()


    batch_size = 1
    #torch.set_grad_enabled(False)
    ct.set_core_number(1)
    ct.set_core_version('MLU270')
    ct.save_as_cambricon('yolov3_270_'+str(batch_size))

    trace_input = torch.randn(1, 3, imgsz[0], imgsz[1], dtype=torch.float)
    
    input_mlu_data = trace_input.to(ct.mlu_device())
    quantized_model_traced = torch.jit.trace(quantized_model, input_mlu_data, check_trace = False)      
    example = torch.randn(batch_size, 3, imgsz[0], imgsz[1]).float()
    quantized_model_traced(example.to(device))
    print("finished")

if __name__ == '__main__':
    with torch.no_grad():
        #quantization_model()
        #val_model()
        #val_model_mlu()
        generate_onfine()