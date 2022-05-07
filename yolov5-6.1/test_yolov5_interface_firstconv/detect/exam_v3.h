#ifndef _EXAM_DETECT__H_
#define _EXAM_DETECT__H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cnrt.h>

struct Box {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	int classes;
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
	int dev_id = 0;
	int dev_channel = -1;
	cnrtInvokeParam_t 		invokeParam;				//invoke参数
	int init = false;
	
	float base_threhold = 0.3f;
	float classify_threhold[3] = {base_threhold, base_threhold, base_threhold};
	float iou_threhold = 0.5f;
	
	const int input_size_w = 640;
	const int input_size_h = 640;
	std::vector<float> stdvs = {1.0f/255, 1.0f/255, 1.0f/255};
	

	
public:
	int batch_size = 1;
	const int classify_number = 3;
	std::string OBJ_LABELS[3] = {"head","safehat","person"};
	EXAMDetect(std::string models_path);

	void Detect(cv::Mat& img, std::vector<Box>&exams);
	
	void set_conf(float base_conf, std::vector<float> clas_confs);
	void set_nms(float nms);
	
	void release();
	
	
	~EXAMDetect();
	
};
#endif 