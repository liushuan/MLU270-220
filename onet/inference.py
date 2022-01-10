import os
import torch
import cv2
import numpy as np

import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct


from net import ONET
import time

ct.set_core_number(1)
ct.set_core_version("MLU270")
torch.set_grad_enabled(False)


def transform(img):
    img = img.astype('float32')
    img = img - (104, 117, 124)
    #img = img - (124, 117, 104)
    #img = img - 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    return img
    
    
def transform2(img):
    img = img.astype('float32')
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    return img
    
def get_image_tensor(img_path):
    Image = cv2.imread(img_path)
    height, width, _ = Image.shape
    if height != 48 or width != 48:
        Image = cv2.resize(Image, (48, 48))
    Image = transform(Image)
    #data = torch.from_numpy(Image).float().unsqueeze(0).to(device)
    data = Image[np.newaxis, :]
    return data


def cpu_infer(net):
    path = "imgs/2.jpg"
    data = get_image_tensor(path)
    data = torch.from_numpy(data)
    t1 = time.time()
    out = net(data)
    t2 = time.time()
    print("out1:", out, "time:",(t2-t1))
    
    input_data=torch.randn((1,3,48,48) ,dtype=torch.float)
    out = net(input_data)
    print("out2:", out)
    
def quantize_model(net):
    print("quantize_model")
    #mean = [104, 117, 124]
    #std = [1/127,1/127,1/127]
    mean = [0, 0, 0]
    std = [1, 1, 1]
    #dtype='int8'
    #'mean':mean, 'std':std, 
    net_quantization = mlu_quantize.quantize_dynamic_mlu(net, {'iteration': 100, 'firstconv':False}, dtype='int8', gen_quant=True)
    
    for i in range(100):
        input_img = get_image_tensor("quan_image/" + str(i)+".jpg")
        input_img = torch.from_numpy(input_img)
        output = net_quantization(input_img)
        
    torch.save(net_quantization.state_dict(), 'onet_quantization.pth')
    print("test net_quantization")
    #input_img=torch.randn((1,3,48,48) ,dtype=torch.float)
    
    input_img = get_image_tensor("imgs/2.jpg")
    input_img = torch.from_numpy(input_img)
 
    output = net_quantization(input_img)
    print("net_quantization output:", output)
 
    return net_quantization

def inference_model():
    print("inference model")
    net = ONET()
    net = mlu_quantize.quantize_dynamic_mlu(net)
    net.load_state_dict(torch.load('onet_quantization.pth'))
    net.eval()
    input_data=torch.randn((1,3,48,48) ,dtype=torch.float)
    input_img = get_image_tensor("imgs/2.jpg")
    input_img = torch.from_numpy(input_img)
    
    #input_img = input_img.type(torch.HalfTensor)
    #a_float = a_half.type(torch.FloatTensor)
    # step 3
    net_mlu = net.to(ct.mlu_device())
    input_mlu = input_img.to(ct.mlu_device())
    input_data_mlu = input_data.to(ct.mlu_device())
    # step 4
    t1 = time.time()
    output=net_mlu(input_mlu)
    t2 = time.time()
    
    output2=net_mlu(input_data_mlu)
    print("inference_model output:",output.cpu(), "time:",(t2-t1))
    print("output2:", output2.cpu())
    
    
    
def fuse_inference_model():
    print("inference model")
    net = ONET()
    net = mlu_quantize.quantize_dynamic_mlu(net)
    net.load_state_dict(torch.load('onet_quantization.pth'))
    net.eval()
    input_data=torch.randn((1,3,48,48) ,dtype=torch.float)
    input_img = get_image_tensor("imgs/2.jpg")
    input_img = torch.from_numpy(input_img)
    
    ct.save_as_cambricon("onet.cambricon")
    trace_model = torch.jit.trace(net.to(ct.mlu_device()),
                                      input_img.to(ct.mlu_device()),
                                      check_trace=False)
    
    t1 = time.time()
    output=trace_model(input_img.to(ct.mlu_device()))
    t2 = time.time()
    print("fuse_inference_model output:",output.cpu(), "time:",(t2-t1))


# main
if __name__ == "__main__":

   
    #ct.set_core_number(1)                 # 设置MLU core number
    #ct.set_core_version("MLU270")                      # 设置MLU core version
    #ct.set_input_format(args.input_format)  

    use_cuda = False
    device = torch.device("cuda:1" if use_cuda else "cpu")
    path = r'./imgs/2_a.jpg'
    net = ONET().to(device)
    net.load_state_dict(torch.load("./det3-half_1.2.pkl", map_location=device), strict=True)
    net.eval()
    cpu_infer(net)
    
    net_quantization = quantize_model(net)
    
    inference_model()
    
    fuse_inference_model()

'''
    if os.path.exists(path):
        data = get_image_tensor(path)
        data = torch.from_numpy(data)
        out = net(data)

        landmarks = out[0]

        img = cv2.imread(path)

        landmarks = landmarks.detach().numpy()[0]
        print("landmarks", landmarks)
        for i in range(5):
            cv2.circle(img, (int((landmarks[2 * i] + 0.5)  * 96), int((landmarks[2 * i + 1] + 0.5)  * 96)), 2, (255, 0, 0))

        landmarks = out[0]
        angle = out[1]

        angle = angle #* 90

        print(landmarks)
        print(angle)

        cv2.imshow("w", img)
        cv2.waitKey(0)
        '''