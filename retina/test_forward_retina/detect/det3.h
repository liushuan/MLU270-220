#ifndef _DET3_NET_H
#define _DET3_NET_H

#include <opencv2/opencv.hpp>
#include <vector>

#include <glog/logging.h>
#include <cnrt.h>

class Det3Net {

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
	
	const int input_size_w = 48;
	const int input_size_h = 48;

	std::vector<float> means = { 127.5, 127.5, 127.5 };
	std::vector<float> stdvs = {0.0078125f, 0.0078125f, 0.0078125f};
	
	
	float threhold = 0.8f;
	
public:
	Det3Net(std::string models_path);
	void init();
	
	bool Detect(cv::Mat& img);
	
	~Det3Net();
	
};
#endif 