
import glob
import cv2
import time


from exam_postprocess.exam_classify import ExamClassify
from exam_postprocess.Interface import InterfaceAnalyse

def test_exam():
    examClassify = ExamClassify("../weights/final_exam_classify.pth", 0)
    path = r'E:\work_space\fcn_exam_classify\exam_postprocess\roi_imgs'
    file_lists = glob.glob(path + r'\**\*.jpg', recursive=True)
    print("file_lists len is:", len(file_lists))
    imgs = []
    for file in file_lists:
        img = cv2.imread(file)
        imgs.append(img)
    result = examClassify.Recognition(imgs)
    print(result)

def test_interface():
    interfaceA = InterfaceAnalyse("../weights/final_exam_classify.pth", 0)

    interfaceA.set_spend_time(16)
    interfaceA.set_continue_time(7)

    path = r'E:\work_space\fcn_exam_classify\exam_postprocess\roi_imgs'
    file_lists = glob.glob(path + r'\**\*.jpg', recursive=True)
    print("file_lists len is:", len(file_lists))
    imgs = []
    for file in file_lists:
        img = cv2.imread(file)
        imgs.append(img)

    status = interfaceA.analyse_imgs(imgs)
    print("status is:", status)

    length = 100
    time1 = time.time()
    for i in range(100):
        status = interfaceA.analyse_imgs(imgs)
    time2 = time.time()
    print("cost time is:", (time2-time1)/length)

if __name__ == "__main__":
    test_interface()