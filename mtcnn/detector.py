import math
import numpy as np
import torch
from model import PNet, RNet, ONet
import time
import cv2

import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct


ct.set_core_number(1)
ct.set_core_version("MLU270")
torch.set_grad_enabled(False)

def transform(img):
    img = img.astype('float32')
    img = img - 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    return img
    
def get_image_tensor(img_path, size_shape):
    Image = cv2.imread(img_path)
    height, width, _ = Image.shape
    if height != size_shape[0] or width != size_shape[1]:
        Image = cv2.resize(Image, size_shape)
    Image = transform(Image)
    #data = torch.from_numpy(Image).float().unsqueeze(0).to(device)
    data = Image[np.newaxis, :]
    return data

def quantize_model():
    print("quantize_model.")
    pnet, rnet, onet= PNet(), RNet(), ONet()
    pnet.eval()
    rnet.eval()
    onet.eval()
    
    input_img1 = get_image_tensor("quan_image/3.jpg", (12, 12))
    input_img1 = torch.from_numpy(input_img1)
    input_img2 = get_image_tensor("quan_image/4.jpg", (24, 24))
    input_img2 = torch.from_numpy(input_img2)
    input_img3 = get_image_tensor("quan_image/5.jpg", (48, 48))
    input_img3 = torch.from_numpy(input_img3)
    pout = pnet(input_img1)
    rout = rnet(input_img2)
    oout = onet(input_img3)
    print("pout:", pout)
    print("rout:", pout)
    print("oout:", pout)
    
    #mean = [104, 117, 124]
    #std = [1/127,1/127,1/127]
    #dtype='int8'
    #'mean':mean, 'std':std, 
    net_quantization_pnet = mlu_quantize.quantize_dynamic_mlu(pnet, {'iteration': 100, 'firstconv':False}, dtype='int8', gen_quant=True)
    net_quantization_rnet = mlu_quantize.quantize_dynamic_mlu(rnet, {'iteration': 100, 'firstconv':False}, dtype='int8', gen_quant=True)
    net_quantization_onet = mlu_quantize.quantize_dynamic_mlu(onet, {'iteration': 100, 'firstconv':False}, dtype='int8', gen_quant=True)
    
    for i in range(100):
        input_img = get_image_tensor("quan_image/" + str(i)+".jpg", (12, 12))
        input_img = torch.from_numpy(input_img)
        output = net_quantization_pnet(input_img)
        
        input_img = get_image_tensor("quan_image/" + str(i)+".jpg", (24, 24))
        input_img = torch.from_numpy(input_img)
        output = net_quantization_rnet(input_img)
        
        input_img = get_image_tensor("quan_image/" + str(i)+".jpg", (48, 48))
        input_img = torch.from_numpy(input_img)
        net_quantization_onet(input_img)
        
       
    torch.save(net_quantization_pnet.state_dict(), 'pnet_quantize.pth')
    torch.save(net_quantization_rnet.state_dict(), 'rnet_quantize.pth')
    torch.save(net_quantization_onet.state_dict(), 'onet_quantize.pth')
    
    output1 = net_quantization_pnet(input_img1)
    output2 = net_quantization_rnet(input_img2)
    output3 = net_quantization_onet(input_img3)
    print("net_quantization_pnet output1:", output1)
    print("net_quantization_pnet output2:", output2)
    print("net_quantization_pnet output3:", output3)
    print("quantize_model finished.")

def generate_offline():
    print("generate_offline.")
    pnet, rnet, onet= PNet(), RNet(), ONet()
    pnet.eval()
    rnet.eval()
    onet.eval()
    
    net_quantization_pnet = mlu_quantize.quantize_dynamic_mlu(pnet)
    net_quantization_pnet.load_state_dict(torch.load('pnet_quantize.pth'))
    
    net_quantization_rnet = mlu_quantize.quantize_dynamic_mlu(rnet)
    net_quantization_rnet.load_state_dict(torch.load('rnet_quantize.pth'))
    
    net_quantization_onet = mlu_quantize.quantize_dynamic_mlu(onet)
    net_quantization_onet.load_state_dict(torch.load('onet_quantize.pth'))
    
    input_img1 = get_image_tensor("quan_image/3.jpg", (12, 12))
    input_img1 = torch.from_numpy(input_img1)
    input_img2 = get_image_tensor("quan_image/4.jpg", (24, 24))
    input_img2 = torch.from_numpy(input_img2)
    input_img3 = get_image_tensor("quan_image/5.jpg", (48, 48))
    input_img3 = torch.from_numpy(input_img3)
    
    ct.save_as_cambricon("pnet.cambricon")
    trace_model_pnet = torch.jit.trace(net_quantization_pnet.to(ct.mlu_device()),input_img1.to(ct.mlu_device()),check_trace=False)
    output1_a, output1_b=trace_model_pnet(input_img1.to(ct.mlu_device()))
    
    ct.save_as_cambricon("rnet.cambricon")
    trace_model_rnet = torch.jit.trace(net_quantization_rnet.to(ct.mlu_device()),input_img2.to(ct.mlu_device()),check_trace=False)
    output2_a, output2_b=trace_model_rnet(input_img2.to(ct.mlu_device()))


    ct.save_as_cambricon("onet.cambricon")
    trace_model_onet = torch.jit.trace(net_quantization_onet.to(ct.mlu_device()),input_img3.to(ct.mlu_device()),check_trace=False)
    output3_a, output3_b, output3_c=trace_model_onet(input_img3.to(ct.mlu_device()))
 
    print("fuse_inference_model output1:",output1_a.cpu(), output1_b.cpu())
    print("fuse_inference_model output2:",output2_a.cpu(),output2_b.cpu())
    print("fuse_inference_model output3:",output3_a.cpu(),output3_a.cpu(), output3_c.cpu())
    
if __name__ == "__main__":
    quantize_model()
    generate_offline()