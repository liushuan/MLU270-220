import argparse
import torch

import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time


import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct

ct.set_core_number(1)
ct.set_core_version("MLU270")
#ct.set_core_version("MLU220")
torch.set_grad_enabled(False)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



def transform(img):
    img = img.astype('float32')
    img = img - (104, 117, 123)
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


def quanza_model():
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    trained_model = r'./weights/mobilenet0.25_Final.pth'
    cpu = True
    net = load_model(net, trained_model, cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    input_img = get_image_tensor("quan_image/10.jpg", (640, 640))
    input_img = torch.from_numpy(input_img)
    output = net(input_img)
    print("output:", output)
    net_quantization = mlu_quantize.quantize_dynamic_mlu(net, {'iteration': 1000, 'firstconv':False}, dtype='int8', gen_quant=True)
    
    for i in range(1000):
        input_img1 = get_image_tensor("quan_image/" + str(i)+".jpg", (640, 640))
        input_img1 = torch.from_numpy(input_img1)
        output = net_quantization(input_img1)
        
        if i % 100 == 0:
            print("curent i:", i)
        
    torch.save(net_quantization.state_dict(), './weights/retina_quantize.pth')
    
    output1 = net_quantization(input_img)
    
    print("output1:", output1)
    
def fuse_quanza_model():
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    trained_model = r'./weights/mobilenet0.25_Final.pth'
    cpu = True
    net = load_model(net, trained_model, cpu)
    net.eval()
    print('Finished loading model!')
    
    net_quantization = mlu_quantize.quantize_dynamic_mlu(net)
    net_quantization.load_state_dict(torch.load('./weights/retina_quantize.pth'))
    print("quantization model load finish")
    input_img1 = get_image_tensor("curve/test2.jpg", (640, 640))
    input_img1 = torch.from_numpy(input_img1)
    
    ct.save_as_cambricon("./weights/retinaface.cambricon")
    trace_model = torch.jit.trace(net_quantization.to(ct.mlu_device()),input_img1.to(ct.mlu_device()),check_trace=False)
    print("generate trace")
    #input_data_b1 =torch.randn((4,3,640,640) ,dtype=torch.float)
    #output1_a, output1_b=trace_model_pnet(input_data_b1.to(ct.mlu_device())) #生成batch
    output1 = trace_model(input_img1.to(ct.mlu_device()))
    print("output1",output1[0].cpu(),output1[1].cpu(),output1[2].cpu())
    return output1[0].cpu(),output1[1].cpu(),output1[2].cpu()
 
def processpst(output1, output2, output3):
    #loc conf landmark
    img_raw = cv2.imread("curve/test2.jpg")
    img = np.float32(img_raw)
    
    resize = 1
    confidence_threshold = 0.02
    vis_thres = 0.6
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    save_image = True
    
    device = torch.device("cpu")
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    
    scale = scale.to(device)
    im_height = 640
    im_width = 640
    cfg = cfg_mnet

    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)


    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(output1.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = output2.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(output3.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    # show image
    if save_image:
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = "output.jpg"
        cv2.imwrite(name, img_raw)
 
    
if __name__ == '__main__':

    #quanza_model()
    out1, out2,out3 = fuse_quanza_model()
    processpst(out1, out2, out3)
    '''
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    trained_model = r'./weights/mobilenet0.25_Final.pth'
    cpu = True
    net = load_model(net, trained_model, cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    resize = 1

    confidence_threshold = 0.02
    vis_thres = 0.6
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    save_image = True
    # testing begin
    for i in range(100):
        image_path = "./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # show image
        if save_image:
            for b in dets:
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)
    '''
