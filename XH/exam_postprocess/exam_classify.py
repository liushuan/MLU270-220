import torch
import os
import cv2
import numpy as np
from .model import EXAMNETTest

class ExamClassify():
    def __init__(self, model_path, gpu_id):
        super(ExamClassify, self).__init__()

        self.size_shape = (128, 128)
        self.batch_size = 50

        if gpu_id >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                print("ERROR device is null.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.numbr_class = 20
        self.net = EXAMNETTest(self.numbr_class).to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval().to(self.device)
        print("net load finished.")

    def transform(self, img):
        height, width, _ = img.shape
        if height != self.size_shape[0] or width != self.size_shape[1]:
            img = cv2.resize(img, self.size_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img /= 255
        img = np.transpose(img, (2, 0, 1))
        return img

    def Recognition(self, origin_images):

        input_imgs = [self.transform(img) for img in origin_images]
        input_imgs = np.stack([img for img in input_imgs], axis=0)
        inputs = torch.from_numpy(input_imgs).to(self.device)
        output = self.net(inputs)
        pred = torch.argmax(output, 1)
        return pred