#ifndef __EXAM_CLASSIFY_H_
#define __EXAM_CLASSIFY_H_

#include <opencv2/opencv.hpp>
#include <vector>

#include <cnrt.h>

using std::string;
using std::vector;

struct Classify{
	int class_id = -1;
	float score = 0;
};

class ExamClassify {

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
	
	const int input_size_w = 128;
	const int input_size_h = 128;

	int output_size = 20;
	
	std::vector<float> means = { 0, 0, 0 };
	std::vector<float> stdvs = {0.00392156f, 0.00392156f, 0.00392156f};

public:

	ExamClassify(string models_path);
	void init();
	int Recognition(cv::Mat & origin_images, Classify & classify);

	~ExamClassify();
};
#endif 