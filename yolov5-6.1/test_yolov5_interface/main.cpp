#include <opencv2/opencv.hpp>
#include <iostream>

#include "testPR.h"
#include "./detect/exam_v3.h"


static void drawimg(cv::Mat &img, std::vector<Box> &exams){
	for(int i = 0; i < exams.size(); i++){
		cv::Rect rect(exams[i].xmin, exams[i].ymin, exams[i].xmax-exams[i].xmin, exams[i].ymax-exams[i].ymin);
		cv::rectangle(img, rect, cv::Scalar(255, 255, 0), 2);
	}
}

void test_exam(){
	
	EXAMDetect examDetect("./weights/yolov5_270_1_1.cambricon");
	
	std::vector<std::string> files, file_names;
	std::string path = "./images";
	util::getFiles(path, files, file_names, ".jpg");
	std::cout<<"file size:"<<files.size()<<std::endl;
	for(int i = 0; i < files.size(); i++){
		cv::Mat img = cv::imread(files[i]);
		std::cout<<"file:"<<files[i]<<std::endl;
		double start = cv::getTickCount();
		std::vector<Box> exams;
		examDetect.Detect(img, exams);
		double end = cv::getTickCount();
		std::cout<<"get exams size is:"<<exams.size()<<" time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
		drawimg(img, exams);
		cv::imwrite("output/"+std::to_string(i)+".jpg", img);
	}
}


static cv::Mat resize_with_crop(cv::Mat origin_image, int input_image_size_w, int input_image_size_h, int &pad_top, int &pad_left, int &resize_w, int &resize_h) {

	float radio = (float)(input_image_size_w) / input_image_size_h;
	cv::Mat  resized_image;
	int h = origin_image.rows;
	int w = origin_image.cols;
	//int pad_left = 0, resize_w = 0, pad_top = 0, resize_h = 0;
	if (h*radio > w) {
		resize_w = int((float)w / h * input_image_size_h);
		cv::resize(origin_image, resized_image, cv::Size(resize_w, input_image_size_h));
		pad_left = (input_image_size_w - resize_w) / 2;
		int pad_wright = (input_image_size_w - resize_w + 1) / 2;
		cv::copyMakeBorder(resized_image, resized_image, 0, 0, pad_left, pad_wright, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	else {
		resize_h = int((float)h / w * input_image_size_w);
		cv::resize(origin_image, resized_image, cv::Size(input_image_size_w, resize_h));
		pad_top = (input_image_size_h - resize_h) / 2;
		int pad_boom = (input_image_size_h - resize_h + 1) / 2;
		cv::copyMakeBorder(resized_image, resized_image, pad_top, pad_boom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	return resized_image;
}


int main(){
	
	test_exam();
	
	getchar();
	
	return 0;
}