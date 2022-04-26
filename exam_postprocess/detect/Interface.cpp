
#include "Interface.h"

InterfaceAnalyse::InterfaceAnalyse(std::string models_path){
	
	examClassify = std::make_shared<ExamClassify>(models_path);
	
}

void InterfaceAnalyse::set_spend_time(int spend_t) {
	exam_analyse.set_spend_time(spend_t);
}

void InterfaceAnalyse::set_continue_time(int continue_t) {
	exam_analyse.set_continue_time(continue_t);
}

bool InterfaceAnalyse::analyse_imgs(std::vector<cv::Mat> &imgs) {
	
	std::vector<int> classifyes;
	std::cout<<"classify:"<<std::endl;
	for(int i = 0; i < imgs.size(); i++){
		Classify classify;
		int status = examClassify->Recognition(imgs[i], classify);
		classifyes.push_back(classify.class_id);
		std::cout<<classify.class_id<<" ";
	}
	std::cout<<std::endl;
	return exam_analyse.analyse(classifyes);

}