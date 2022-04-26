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
	
	EXAMDetect examDetect("./weights/yolov3_270_1_4.cambricon", "./weights/yolov3_270_4_4.cambricon");
	
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

void test_exam_batch(){
	
	EXAMDetect examDetect("./weights/yolov3_270_1_4.cambricon", "./weights/yolov3_270_4_4.cambricon");
	std::vector<cv::Mat> imgs;
	
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
	
	for(int i = 0; i < 10; i++){
		cv::Mat img = cv::imread("imgs/"+std::to_string(i)+".jpg");
		
		imgs.push_back(img);
		
		if (imgs.size() == 4 || i == 9){
			double start = cv::getTickCount();
			std::vector<std::vector<Box>> exams;
			
			examDetect.Detect(imgs, exams);
			double end = cv::getTickCount();
			std::cout<<"batch size is:"<<exams.size()<<" time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
			
			for(int k = 0; k < imgs.size(); k++){
				drawimg(imgs[k], exams[k]);
				cv::imwrite("output_b/"+std::to_string(i)+"_"+std::to_string(k)+".jpg", imgs[k]);
				std::cout<<"detect obj size is:"<< exams[k].size()<<std::endl;
			}
			imgs.clear();
		}
	}
}


#include "testPR.h"
void test_exam_PR()
{
	std::string img_path = "./data/images";
	std::string txt_path = "./data/txt";
	std::vector<util::Target> targets = util::get_targets(txt_path, img_path);
	std::cout << "generate target finished." << std::endl;
	std::vector<cv::Scalar> color_label = {
		cv::Scalar(50, 100, 200),cv::Scalar(100, 50, 200), cv::Scalar(200, 50, 100), cv::Scalar(200, 100, 50), cv::Scalar(50, 200, 100), cv::Scalar(100, 200, 50), 
		cv::Scalar(0, 50, 150),cv::Scalar(0, 150, 50), cv::Scalar(50, 0, 150), cv::Scalar(50, 150, 0), cv::Scalar(150, 0, 50), cv::Scalar(150, 50, 0),
		cv::Scalar(20, 80, 180),cv::Scalar(20, 180, 80), cv::Scalar(80, 20, 180), cv::Scalar(80, 180, 20), cv::Scalar(180, 20, 80), cv::Scalar(180, 80, 20),
		cv::Scalar(80, 160, 240),cv::Scalar(80, 240, 160), cv::Scalar(160, 80, 240), cv::Scalar(160, 240, 80), cv::Scalar(240, 80, 160), cv::Scalar(240, 160, 80)
	};

	EXAMDetect examDetect("./weights/yolov3_270_1_4.cambricon", "./weights/yolov3_270_4_4.cambricon");
	float view_threhold = 0.3f;
	std::vector<util::Target> preds(targets.size());
	for (size_t i = 0; i < targets.size(); i++)
	{
		util::Target t = targets[i];
		cv::Mat img = cv::imread(t.img_name);
		double start = cv::getTickCount();
		std::vector<Box> vanc_boxs;
		examDetect.Detect(img, vanc_boxs);
		double end = cv::getTickCount();
		if (i < 20){
			std::cout<<"cost time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
		}
		//检测结果画图
		for (size_t j = 0; j < vanc_boxs.size(); j++)
		{
			if (vanc_boxs[j].score >= view_threhold) {
				cv::Rect rect(vanc_boxs[j].xmin, vanc_boxs[j].ymin, vanc_boxs[j].xmax-vanc_boxs[j].xmin, vanc_boxs[j].ymax-vanc_boxs[j].ymin);
				cv::rectangle(img, rect, color_label[vanc_boxs[j].classes], 2);
				cv::putText(img, examDetect.OBJ_LABELS[vanc_boxs[j].classes] + ":" + std::to_string(vanc_boxs[j].score), cv::Point(rect.x + 5, rect.y + rect.height / 2), 1, 1.4, color_label[vanc_boxs[j].classes], 2);
			}
		}
		cv::imwrite("./result/" + std::to_string(i)+".jpg", img);

		preds[i].img_name = t.img_name;
		preds[i].txt_name = t.txt_name;
		for (size_t j = 0; j < vanc_boxs.size(); j++)
		{
			util::Obj obj;
			obj.classify = vanc_boxs[j].classes;
			obj.match = false;
			obj.score = vanc_boxs[j].score;
			obj.xmin = vanc_boxs[j].xmin;
			obj.ymin = vanc_boxs[j].ymin;
			obj.xmax = vanc_boxs[j].xmax;
			obj.ymax = vanc_boxs[j].ymax;
			preds[i].objs.push_back(obj);
		}
		if (i % 50 == 0) {
			std::cout << "inference file count is:" << i << std::endl;
		}
	}
	
	util::test_PR(preds, targets);
	std::cout << "test finished." << std::endl;
	getchar();
	
}




int main(){
	
	//test_exam();
	test_exam_batch();
	//test_exam_PR();
	
	return 0;
}