#include <opencv2/opencv.hpp>
#include <iostream>

#include "./detect/retina_face.h"
#include "./detect/headpose.h"
#include "./detect/det3.h"

#include "task/trackTask.h"
#include "decoder/decode_task.h"


static void drawimg(cv::Mat &img, std::vector<FaceA> &face){
	for(int i = 0; i < face.size(); i++){
		cv::rectangle(img, face[i].rect, cv::Scalar(255, 255, 0), 2);
		for(int j = 0; j < 5; j++){
			cv::circle(img, cv::Point(face[i].landmarks[2*j], face[i].landmarks[2*j+1]), 5, cv::Scalar(255,255,0), -1);
		}
	}
	
}
void test_retina(){
	RetinaFace retina("./weights/retinaface.cambricon.cambricon");
	
	for(int i = 0; i < 10; i++){
		cv::Mat img = cv::imread("imgs/"+std::to_string(i)+".jpg");
	
		double start = cv::getTickCount();
		std::vector<FaceA> faces;
		retina.Detect(img, faces);
		double end = cv::getTickCount();
		std::cout<<"get face size is:"<<faces.size()<<" time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
		drawimg(img, faces);
		cv::imwrite("output_"+std::to_string(i)+".jpg", img);
	}
}

void test_head(){
	
	HEADPose headpose("./weights/headpose.cambricon.cambricon");
	for(int i = 0; i < 10; i++){
		cv::Mat img = cv::imread("imgs/2_a.jpg");
		double start = cv::getTickCount();
		cv::resize(img, img, cv::Size(48,48));
		
		Pose pose = headpose.Detect(img);
		double end = cv::getTickCount();
		std::cout<<"headpose time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
	}
}

void test_det3(){
	
	Det3Net det3Net("./weights/det3.cambricon.cambricon");
	for(int i = 0; i < 10; i++){
		cv::Mat img = cv::imread("imgs/2.jpg");
		double start = cv::getTickCount();
		cv::resize(img, img, cv::Size(48,48));
		bool value = det3Net.Detect(img);
		double end = cv::getTickCount();
		std::cout<<"det3 time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
	}
}

void trackStartCB(void * pUser, std::vector<YourStruct1> &track_starts) {
	for (size_t i = 0; i < track_starts.size(); i++)
	{
		std::cout << "found face track id:" << track_starts[i].head_id <<" direct:"<< track_starts[i].direct <<" time:"<<track_starts[i].time <<
			" rect.x:"<< track_starts[i].rect.x<<" "<< track_starts[i].rect.y<<" "<< track_starts[i].rect.width <<" "<< track_starts[i].rect.height<< std::endl;
		cv::Mat head = track_starts[i].img(track_starts[i].rect);

	}
}

void trackEndCB(void * pUser, std::vector<YourStruct2> &track_ends) {
	for (size_t i = 0; i < track_ends.size(); i++)
	{
		std::cout << "track face miss id:" << track_ends[i].head_id << " direct:" << track_ends[i].direct << " time:" << track_ends[i].time << std::endl;
		if (!track_ends[i].face.empty()){
			cv::imwrite("output/" + track_ends[i].time+"_"+std::to_string(i)+".jpg", track_ends[i].face);
		}
	}
}

void test_face_track(){
	int decoder_device_id = -1;
	int fps = 30;
	std::string face_engin_name = "./weights/retinaface.cambricon.cambricon";
	std::string head_engin_name = "./weights/headpose.cambricon.cambricon";
	std::string det3_engin_name = "./weights/det3.cambricon.cambricon";
	RetinaFace faceDetect(face_engin_name);
	HEADPose headpose(head_engin_name);
	Det3Net det3net(det3_engin_name);
	//test_head();
	
	
	std::shared_ptr<TrackTask> trackTaskN;
	std::shared_ptr<DecodeTask> decoderTaskN;

	//std::vector<cv::Point> points = {cv::Point(50, 50), cv::Point(1500, 50), cv::Point(1500, 1000) , cv::Point(50, 1000) };
	//std::vector<int> directs = {DIRECT_OUT, DIRECT_IN, DIRECT_IN , DIRECT_IN};

	trackTaskN = std::make_shared<TrackTask>(&faceDetect, &headpose, &det3net);
	//trackTaskN->setTrackPoints(points, directs);
	trackTaskN->set_start_cb(trackStartCB);
	trackTaskN->set_end_cb(trackEndCB);

	trackTaskN->set_show(true);
	trackTaskN->set_screen(1280, 720);

	decoderTaskN = std::make_shared<DecodeTask>("b.mp4", trackTaskN.get(), decoder_device_id);
	decoderTaskN->set_fps(fps);
	
	trackTaskN->start();
	decoderTaskN->start();
	
	std::cout<<"enter any key to stop. "<<std::endl;
	getchar();
	getchar();
	decoderTaskN->stop();
	trackTaskN->stop();
	
}


#include "testPR.h"
void test_face_pr()
{
	std::string img_path = "./test/images";
	std::string txt_path = "./test/txt";
	std::vector<util::Target> targets = util::get_targets(txt_path, img_path);
	std::cout << "generate target finished." << std::endl;
	std::vector<cv::Scalar> color_label = {
		cv::Scalar(50, 100, 200)
	};
	RetinaFace retina("./weights/retinaface_512.cambricon.cambricon");
	
	float view_threhold = 0.3f;
	std::vector<util::Target> preds(targets.size());
	for (size_t i = 0; i < targets.size(); i++)
	{
		util::Target t = targets[i];
		cv::Mat img = cv::imread(t.img_name);
		double start = cv::getTickCount();
		std::vector<FaceA> vanc_boxs;
		retina.Detect(img, vanc_boxs);
		double end = cv::getTickCount();
		if (i < 20){
			std::cout<<"cost time is:"<<(end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
		}
		//检测结果画图
		for (size_t j = 0; j < vanc_boxs.size(); j++)
		{
			if (vanc_boxs[j].score >= view_threhold) {
				cv::Rect rect = vanc_boxs[j].rect;
				cv::rectangle(img, rect, color_label[0], 2);
			}
		}
		cv::imwrite("./result/" + std::to_string(i)+".jpg", img);

		preds[i].img_name = t.img_name;
		preds[i].txt_name = t.txt_name;
		for (size_t j = 0; j < vanc_boxs.size(); j++)
		{
			util::Obj obj;
			obj.classify = 0;
			obj.match = false;
			obj.score = vanc_boxs[j].score;
			obj.xmin = vanc_boxs[j].rect.x;
			obj.ymin = vanc_boxs[j].rect.y;
			obj.xmax = vanc_boxs[j].rect.x + vanc_boxs[j].rect.width;
			obj.ymax = vanc_boxs[j].rect.y + vanc_boxs[j].rect.height;
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
	
	test_head();
	test_det3();
	test_face_pr();
	
	//test_face_track();
	
	return 0;
}
