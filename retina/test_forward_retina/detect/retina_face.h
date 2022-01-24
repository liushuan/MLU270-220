#ifndef _FACE_DETECT_RETINA_H_
#define _FACE_DETECT_RETINA_H_

#include <opencv2/opencv.hpp>
#include <vector>

#include <cnrt.h>

struct RetinaAnchor {
	float anchor[4];
};

struct FaceA {
	cv::Rect2f rect;
	float score;
	float landmarks[10];
	bool operator<(const FaceA &t) const {
		return score < t.score;
	}
	bool operator>(const FaceA &t) const {
		return score > t.score;
	}
};

class RetinaFace {

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
	
	const float threhold = 0.7f;
	const int min_width = 32;
	const float iou_threhold = 0.4f;
	const float fradio_threadhold = 0.4f;
	const float default_score = 0.1f;
	const int input_size_w = 640;
	const int input_size_h = 640;
	std::vector<RetinaAnchor> myAnchors;
	std::vector<int> stride_list = { 8, 16, 32 };
	std::vector<int> minSizes = { 16, 32, 64, 128, 256, 512 };
	std::vector<float> means = { 104, 117, 123 };
	void generate_anchor();

public:
	int batch_size = 1;
	RetinaFace(std::string models_path);
	
	void init();
	
	void sure_face(std::vector<FaceA>&faces);
	
	void Detect(cv::Mat& img, std::vector<FaceA>&faces);
	
	~RetinaFace();
	
};
#endif 