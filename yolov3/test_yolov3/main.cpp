#include <opencv2/opencv.hpp>
#include <iostream>

#include "./detect/exam_v3.h"


static void drawimg(cv::Mat &img, std::vector<Box> &exams){
	for(int i = 0; i < exams.size(); i++){
		cv::Rect rect(exams[i].xmin, exams[i].ymin, exams[i].xmax-exams[i].xmin, exams[i].ymax-exams[i].ymin);
		cv::rectangle(img, rect, cv::Scalar(255, 255, 0), 2);
	}
}

void test_exam(){
	
	EXAMDetect examDetect("./weights/yolov3_270_1.cambricon");
	
	for(int i = 0; i < 10; i++){
		cv::Mat img = cv::imread("imgs/"+std::to_string(i)+".jpg");
	
		double start = cv::getTickCount();
		std::vector<Box> exams;
		examDetect.Detect(img, exams);
		double end = cv::getTickCount();
		std::cout<<"get exams size is:"<<exams.size()<<" time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
		drawimg(img, exams);
		cv::imwrite("output/"+std::to_string(i)+".jpg", img);
	}
}

int main(){
	
	test_exam();
	return 0;
}