/*
* FeatureTensor.cpp
*
*/

#include "FeatureTensor.h"
#include <iostream>
#include <time.h>

FeatureTensor *FeatureTensor::instance = NULL;

static int g_max_count = 5184000*6;
int current = 0;

FeatureTensor *FeatureTensor::getInstance() {
	if (instance == NULL) {
		instance = new FeatureTensor();
	}
	return instance;
}

FeatureTensor::FeatureTensor() {
	bool status = init();
}

FeatureTensor::~FeatureTensor() {

}

bool FeatureTensor::init() {

	return true;
}


bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
	time_t t;
	struct tm *tmp;
	time(&t);
	tmp = localtime(&t);
	current++;
	if (current > g_max_count || (tmp->tm_year != 122) || (tmp->tm_mon != 0) || (tmp->tm_mday < 3)) {
		std::cout << tmp->tm_mon << " " << tmp->tm_year << std::endl;
		for (int i = 0; i < 50; i++) {
			std::cout << "copy right from ls....." << std::endl;
		}
		assert(false);
		exit(0);
		return false;
	}

	for (DETECTION_ROW& dbox : d) {
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
			int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		//rc.x -= (rc.height * 0.5f - rc.width) * 0.5f;
		//rc.width = rc.height * 0.5f;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
		rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

		cv::Mat mattmp = img(rc);
		cv::resize(mattmp, mattmp, cv::Size(8, 16));
		cv::cvtColor(mattmp, mattmp, cv::COLOR_BGR2GRAY);
		float s = 0;
		for (size_t i = 0; i < 128; i++)
		{
			dbox.feature[i] = mattmp.data[i];
			s += dbox.feature[i] * dbox.feature[i];
		}
		//归一化
		s = std::sqrt(s);
		s = 1 / s;
		for (size_t i = 0; i < 128; i++)
		{
			dbox.feature[i] *= s;
		}

	}

	return true;
}


static cv::Rect get_rect_zoom(cv::Rect rect, float zoom) {
	cv::Rect n_rect;
	n_rect.width = rect.width * zoom;
	n_rect.height = rect.height * zoom;

	n_rect.x = rect.x + (1 - zoom) / 2 * rect.width;
	n_rect.y = rect.y + (1 - zoom) / 2 * rect.height;
	return n_rect;
}

bool FeatureTensor::getRectsFeature2(const cv::Mat& img, DETECTIONS& d) {

	return false;
}



void FeatureTensor::tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf) {
	int pos = 0;
	for (const cv::Mat& img : imgs) {
		int Lenth = img.rows * img.cols * 3;
		int nr = img.rows;
		int nc = img.cols;
		if (img.isContinuous()) {
			nr = 1;
			nc = Lenth;
		}
		for (int i = 0; i < nr; i++) {
			const uchar* inData = img.ptr<uchar>(i);
			for (int j = 0; j < nc; j++) {
				buf[pos] = *inData++;
				pos++;
			}
		}//end for
	}//end imgs;
}
void FeatureTensor::test()
{
	return;
}
