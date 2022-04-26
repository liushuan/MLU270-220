#include "detect/exam_classify.h"

#include "detect/Interface.h"
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

std::vector<std::string> getFiles(std::string cate_dir)
{
	std::vector<std::string> files;//存放文件名
	DIR *dir;
	struct dirent *ptr;
	char base[1000];

	if ((dir = opendir(cate_dir.c_str())) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
			continue;
		else if (ptr->d_type == 8)    ///file
									  //printf("d_name:%s/%s\n",basePath,ptr->d_name);
			files.push_back(ptr->d_name);
		else if (ptr->d_type == 10)    ///link file
									   //printf("d_name:%s/%s\n",basePath,ptr->d_name);
			continue;
		else if (ptr->d_type == 4)    ///dir
		{
			files.push_back(ptr->d_name);
			/*
			memset(base,'\0',sizeof(base));
			strcpy(base,basePath);
			strcat(base,"/");
			strcat(base,ptr->d_nSame);
			readFileList(base);
			*/
		}
	}
	closedir(dir);

	//排序，按从小到大排序
	sort(files.begin(), files.end());
	return files;
}


void test_acc(){
  
  std::vector<std::string> pre_class_name = {"00_idle","01_write","02_vacant","03_lpeep","04_rpeep","05_bpeep","06_signal","07_handsdown","08_handspeep","09_handpeep",
                  "10_getitem","11_changeitem","12_opitem","13_sleep","14_standup","15_handsup","16_drinkwater","17_destroypaper","18_turnpaper","19_stretch"};
  
  	//初始化接口文件
	ExamClassify interface_hp("weights/exam_classify_1_1.cambricon.cambricon");
  std::string base_path = "/opt/cambricon/fcn_exam_classify/quan_img/";
  float all_acc[20];
  for  (int i = 0 ; i < pre_class_name.size(); i++){
      std::string sub_dir = base_path + pre_class_name[i] + "/" + pre_class_name[i];
      std::vector<std::string> file_names = getFiles(sub_dir);
      std::cout<<"file:"<<pre_class_name[i] <<" length is: "<<file_names.size()<<std::endl;
      float acc = 0;
     	for (size_t j = 0;  j< file_names.size(); j++)
      {
		      cv::Mat img = cv::imread(sub_dir+"/"+file_names[j]);
		      Classify classify;
		      int status = interface_hp.Recognition(img, classify);
  	      if (status == 0){
            if (classify.class_id == i){
              acc += 1;
            }
          }else{
            std::cout<<"mlu error"<<std::endl;
          }	
	    }
      all_acc[i] = acc / file_names.size();
  }
  
  for(int i = 0; i < 20; i++){
    std::cout<<pre_class_name[i] <<":"<<all_acc[i]<<std::endl;
  }
  
}



void test_exam_classify(){
	
	//初始化接口文件
	ExamClassify interface_hp("weights/exam_classify_1_1.cambricon.cambricon");

	std::vector<std::string> file_names = getFiles("./img");


	for (size_t i = 0; i < file_names.size(); i++)
	{
		cv::Mat img = cv::imread("./img/"+file_names[i]);
		std::cout<<"file name is:"<<file_names[i]<<std::endl;
		
		double start = cv::getTickCount();
		Classify classify;
		int status = interface_hp.Recognition(img, classify);
		double end = cv::getTickCount();
		std::cout << "cost time is:" << (end - start) / cv::getTickFrequency() * 1000 << " ms" <<" class id::::"<<classify.class_id<<":::: score:"<<classify.score<< std::endl;
		
	}
	
}


void test_interface(){
	
	InterfaceAnalyse interfaceAnalyse("weights/exam_classify_1_4.cambricon.cambricon");
	interfaceAnalyse.set_spend_time(20);
	interfaceAnalyse.set_continue_time(6);
	
	std::vector<std::string> file_names = getFiles("./roi_imgs");
	std::vector<cv::Mat> roi_imgs;
	for(int i = 0; i< file_names.size(); i++){
		
		cv::Mat img = cv::imread("./roi_imgs/"+file_names[i]);
    cv::resize(img, img, cv::Size(128,128));
		roi_imgs.push_back(img);
	}
	std::cout<<"roi imgs length is:"<<roi_imgs.size()<<std::endl;

	bool status = interfaceAnalyse.analyse_imgs(roi_imgs);
	
	std::cout<<"result status is:"<< status <<std::endl;
	
}


int main(int argc, char** argv) {
	
	test_exam_classify();
	
	test_interface();
	
  test_acc();
 
	return 0;
}
