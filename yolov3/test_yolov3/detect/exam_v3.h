#ifndef _EXAM_DETECT__H_
#define _EXAM_DETECT__H_

#include <opencv2/opencv.hpp>
#include <vector>

#include <cnrt.h>

struct Box {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	float classes;
};

class EXAMDetect {

private:
	cnrtDev_t dev;
	cnrtModel_t model;
	cnrtFunction_t function;
	cnrtRuntimeContext_t rt_ctx_;
	int inputNum, outputNum;
	int64_t* inputSizeS = nullptr;
    int64_t* outputSizeS = nullptr;
	cnrtDataType_t* input_data_type = nullptr;
    cnrtDataType_t* output_data_type = nullptr;
	
	void** inputCpuPtrS = nullptr;
	void** outputCpuPtrS = nullptr;
	void** outTransCpuPtrS = nullptr;
	void** param = nullptr;
    std::vector<int> in_count;
    std::vector<int> out_count;
	
	void** inputMluPtrS = nullptr;
	void** outputMluPtrS = nullptr;
	
	cnrtQueue_t cnrt_queue;
	
	const float threhold = 0.2f;
	const float iou_threhold = 0.5f;
	
	const int input_size_w = 512;
	const int input_size_h = 512;
	std::vector<float> stdvs = {1.0f/255, 1.0f/255, 1.0f/255};

public:
	int batch_size = 1;
	EXAMDetect(std::string models_path);
	void init();
	
	void Detect(cv::Mat& img, std::vector<Box>&exams);
	
	~EXAMDetect();
	
};
#endif 