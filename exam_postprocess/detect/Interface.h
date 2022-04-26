#ifndef _INTERFACE_H_H
#define _INTERFACE_H_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "exam_classify.h"
#include "exam_analyse.h"

class InterfaceAnalyse {

private:

	ExamAnalyse exam_analyse;
	std::shared_ptr<ExamClassify> examClassify;
	
public:

	InterfaceAnalyse(std::string models_path);

	bool analyse_imgs(std::vector<cv::Mat> &imgs);
	void set_spend_time(int spend_t);
	void set_continue_time(int continue_t);
};


#endif