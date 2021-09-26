import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import os
import torch_mlu
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct

import numpy as np
def get_boxes(prediction, batch_size=1, img_size=640):
    """
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    reshape_value = torch.reshape(prediction, (-1, 1))

    num_boxes_final = reshape_value[0].item()
    print('num_boxes_final: ',num_boxes_final)
    all_list = [[] for _ in range(batch_size)]
    for i in range(int(num_boxes_final)):
        batch_idx = int(reshape_value[64 + i * 7 + 0].item())
        if batch_idx >= 0 and batch_idx < batch_size:
            bl = reshape_value[64 + i * 7 + 3].item()
            br = reshape_value[64 + i * 7 + 4].item()
            bt = reshape_value[64 + i * 7 + 5].item()
            bb = reshape_value[64 + i * 7 + 6].item()

            if bt - bl > 0 and bb -br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                # all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 1].item())

    output = [np.array(all_list[i]).reshape(-1, 6) for i in range(batch_size)]
    # outputs = [torch.FloatTensor(all_list[i]).reshape(-1, 6) for i in range(batch_size)]
    return output
    # jdict = []
    # for si, pred in enumerate(output):
    #     box = pred[:, :4]  #x1, y1, x2, y2
    #     for di, d in enumerate(pred):
    #         box_temp = []
    #         box_temp.append(np.round(box[di][0], 3).item())
    #         box_temp.append(np.round(box[di][1], 3).item())
    #         box_temp.append(np.round(box[di][2], 3).item())
    #         box_temp.append(np.round(box[di][3], 3).item())
    #         jdict.append({'bbox': box_temp, 'score': (np.round(d[5], 5)).item()})
    # sorted_jdict = sorted(jdict, key=lambda x:x['score'], reverse=True)
    # return sorted_jdict

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    global quantized_model
    global quantized_net

    if opt.cfg == 'qua':
        qconfig = {'iteration':2,'firstconv':False}
        quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant=True)
    
    elif opt.cfg == 'mlu':
        from models.yolo import Model

        model = Model('./models/yolov5s.yaml').to(torch.device('cpu'))
        model.float().fuse().eval()

        quantized_net = torch_mlu.core.mlu_quantize.quantize_dynamic_mlu(model)

        state_dict = torch.load("./yolov5s_int8.pt")
        quantized_net.load_state_dict(state_dict, strict=False)

        quantized_net.eval()
        quantized_net.to(ct.mlu_device())

        if opt.jit:
            print("### jit")
            ct.save_as_cambricon('yolov5s_int8_1_4')
            torch.set_grad_enabled(False)
            ct.set_core_number(4)
            trace_input = torch.randn(1, 3, 640, 640, dtype=torch.float)
            input_mlu_data = trace_input.type(torch.HalfTensor).to(ct.mlu_device())
            quantized_net = torch.jit.trace(quantized_net, input_mlu_data, check_trace = False)
        
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        img = torch.cat([img, img, img, img, img, img, img, img], 0)
        
        
        print("path:", path)
        print("img:", img.shape)
        print("im0s:", im0s.shape)

        # Inference
        t1 = time_synchronized()
        
        if opt.cfg == 'qua':
            pred = quantized_model(img)[0]
            torch.save(quantized_model.state_dict(), 'yolov5s_int8.pt')
            print('run qua')
        
        elif opt.cfg == 'mlu':
        
            start_t = time.time()
            img = img.type(torch.HalfTensor).to(ct.mlu_device())
            img = img.to(ct.mlu_device())
            pred = quantized_net(img)[0]
            pred=pred.data.cpu().type(torch.FloatTensor)
            end_t = time.time()
            
            box_result = get_boxes(pred)
            print("im0s.shape:",im0s.shape)
            print(box_result)                  
            res = box_result[0].tolist()
            
            with open("yolov5s_mlu_output.txt","w+") as f:
                for pt in sorted(res, key=lambda x:(x[0],x[1])):
                    f.write("{}\n{}\n{}\n{}\n".format(pt[0],pt[1],pt[2],pt[3]))                  
                    cv2.rectangle(im0s, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)                
                cv2.imwrite("mlu_out_{}.jpg".format(os.path.basename(path).split('.')[0]), im0s)   
            print('run mlu, inference .................time is: ', (end_t - start_t))

        elif opt.cfg == 'cpu':
            pred = model(img, augment=opt.augment)[0]
            print('run cpu')
        
        if opt.cfg != 'cpu':
            continue
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', default='cpu', help='qua and off')
    parser.add_argument('--jit', type=bool,default=False)
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
