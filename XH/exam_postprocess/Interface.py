
from .exam_classify import ExamClassify
from .exam_analyse import ExamAnalyse

class InterfaceAnalyse():
    def __init__(self, model_path, gpu_id):
        super(InterfaceAnalyse, self).__init__()
        self.model = ExamClassify(model_path, gpu_id)
        self.exam_analyse = ExamAnalyse()



    def analyse_imgs(self, imgs):
        classifyes = self.model.Recognition(imgs)
        #print(classifyes)
        return self.exam_analyse.analyse(classifyes)
    def set_spend_time(self, spend_t):
        self.exam_analyse.set_spend_time(spend_t)
    def set_continue_time(self, continue_t):
        self.exam_analyse.set_continue_time(continue_t)