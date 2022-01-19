#ifndef  _FEATURETONSOR_H
#define _FEATURETONSOR_H

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"


#include "model.h"
#include "dataType.h"
typedef unsigned char uint8;

class FeatureTensor
{
public:
	static FeatureTensor* getInstance();
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);
	bool getRectsFeature2(const cv::Mat& img, DETECTIONS& d);
private:
	FeatureTensor();
	FeatureTensor(const FeatureTensor&);
	FeatureTensor& operator = (const FeatureTensor&);
	static FeatureTensor* instance;
	bool init();
	~FeatureTensor();

	void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

	int feature_dim;
public:
	void test();
};

#endif // ! _FEATURETONSOR_H