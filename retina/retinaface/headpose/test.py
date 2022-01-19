import os
import torch
import cv2
import numpy as np

from net import ONET

import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
ct.set_core_number(1)
ct.set_core_version("MLU270")
#ct.set_core_version("MLU220")
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
def get_image_tensor(img_path):
    Image = cv2.imread(img_path)
    height, width, _ = Image.shape
    if height != 48 or width != 48:
        Image = cv2.resize(Image, (48, 48))
    Image = transform(Image)
    #data = torch.from_numpy(Image).float().unsqueeze(0).to(device)
    data = Image[np.newaxis, :]
    return data

def quanza_model():
    use_cuda = False
    device = torch.device("cuda:1" if use_cuda else "cpu")
    net = ONET().to(device)
    net.load_state_dict(torch.load("./headpose.pth", map_location=device), strict=True)
    net.eval()
    print('Finished loading model!')
    #print(net)
    
    input_img = get_image_tensor("imgs/2_a.jpg")
    input_img = torch.from_numpy(input_img)
    output = net(input_img)
    print("output:", output)
    
    net_quantization = mlu_quantize.quantize_dynamic_mlu(net, {'iteration': 1000, 'firstconv':False}, dtype='int8', gen_quant=True)
    
    for i in range(1000):
        input_img1 = get_image_tensor("quan_image/" + str(i)+".jpg")
        input_img1 = torch.from_numpy(input_img1)
        output = net_quantization(input_img1)
        if i % 100 == 0:
            print("curent i:", i)
        
    torch.save(net_quantization.state_dict(), 'headpose_quantize.pth')
    
    output1 = net_quantization(input_img)
    
    print("output1:", output1)

def fuse_quanza_model():
    use_cuda = False
    device = torch.device("cuda:1" if use_cuda else "cpu")
    net = ONET().to(device)
    net.load_state_dict(torch.load("./headpose.pth", map_location=device), strict=True)
    net.eval()
    print('Finished loading model!')
    
    net_quantization = mlu_quantize.quantize_dynamic_mlu(net)
    net_quantization.load_state_dict(torch.load('headpose_quantize.pth'))
    print("quantization model load finish")
    
    input_img1 = get_image_tensor("imgs/2_a.jpg")
    input_img1 = torch.from_numpy(input_img1)
    
    ct.save_as_cambricon("headpose.cambricon")
    trace_model = torch.jit.trace(net_quantization.to(ct.mlu_device()),input_img1.to(ct.mlu_device()),check_trace=False)
    print("generate trace")
    #input_data_b1 =torch.randn((4,3,640,640) ,dtype=torch.float)
    #output1_a, output1_b=trace_model_pnet(input_data_b1.to(ct.mlu_device())) #生成batch
    output1 = trace_model(input_img1.to(ct.mlu_device()))
    print("output1",output1[0].cpu(),output1[1].cpu())


# main
if __name__ == "__main__":

    quanza_model()
    fuse_quanza_model();
    '''
    use_cuda = False
    device = torch.device("cuda:1" if use_cuda else "cpu")

    path = r'./imgs/2_a.jpg'

    net = ONET().to(device)
    net.load_state_dict(torch.load("./o_net_half_test.pkl", map_location=device), strict=True)
    net.eval()

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