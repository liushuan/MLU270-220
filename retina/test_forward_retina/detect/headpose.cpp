#include "headpose.h"

//DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");

//extern int FLAGS_mludevice;
HEADPose::HEADPose(std::string models_path) {
	
	cnrtInit(0);
	std::cout<<"init"<<std::endl;
	unsigned devNum;
	cnrtGetDeviceCount(&devNum);
	std::cout<<"devNum:"<<devNum<<std::endl;
	/*if (FLAGS_mludevice >= 0) {
		CHECK_NE(devNum, 0) << "No device found";
		CHECK_LE(FLAGS_mludevice, devNum) << "valid device count: " << devNum;
	} else {
		LOG(FATAL) << "Invalid device number";
	}

	cnrtGetDeviceHandle(&dev, FLAGS_mludevice);*/
	
	//cnrtSetCurrentDevice(dev);
	
	// 2. load model and get function

	//int size;
	//cnrtGetModelSize(models_path.c_str(), &size);
	std::cout<<"init cnrtLoadModel"<<std::endl;
	cnrtLoadModel(&model, models_path.c_str());
	std::cout<<"start init"<<std::endl;
	init();

	std::cout<< "load model finished."<<std::endl;
}

void HEADPose::init(){
	cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, "subnet0");
    cnrtCreateRuntimeContext(&rt_ctx_, function, NULL);
	
    // 3. get function's I/O DataDesc
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
    cnrtGetInputDataType(&input_data_type, &inputNum, function);
    cnrtGetOutputDataType(&output_data_type, &outputNum, function);
	
    // 4. allocate I/O data space on CPU memory and prepare Input data
	inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
	outTransCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    param = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));

    for (int i = 0; i < inputNum; i++) {
      int ip = inputSizeS[i] / cnrtDataTypeSize(input_data_type[i]);
      auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
	  //get_input_data("imgs/test2.jpg", databuf);
      in_count.push_back(ip);
      inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);  // NHWC
    }

    for (int i = 0; i < outputNum; i++) {
      int op = outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
      float* outcpu = reinterpret_cast<float*>(malloc(op * sizeof(float)));
      out_count.push_back(op);
      outputCpuPtrS[i] = reinterpret_cast<void*>(outcpu);
	  
	  float* outcpu1 = reinterpret_cast<float*>(malloc(op * sizeof(float)));
	  outTransCpuPtrS[i] = reinterpret_cast<void*>(outcpu1);
    }
	// 5. allocate I/O data space on MLU memory and copy Input data
    // Only 1 batch so far
    inputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
    outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
    for (int i = 0; i < inputNum; i++) {
      cnrtMalloc(&inputMluPtrS[i], inputSizeS[i]);
    }
    for (int i = 0; i < outputNum; i++) {
      cnrtMalloc(&outputMluPtrS[i], outputSizeS[i]);
    }
    for (int i = 0; i < inputNum; i++) {
      param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; i++) {
      param[inputNum + i] = outputMluPtrS[i];
    }
	 // 6. create cnrt_queue
    cnrtCreateQueue(&cnrt_queue);
    //cnrtSetRuntimeContextDeviceId(rt_ctx_, dev);  //Dev ordinal value error
	cnrtInitRuntimeContext(rt_ctx_, NULL);
}


Pose HEADPose::Detect(cv::Mat& img) {
	//1. process
	Pose pose;
	float * data = reinterpret_cast<float*>(inputCpuPtrS[0]);
	int i = 0;
	for (int row = 0; row < input_size_h; ++row) {
		uchar* uc_pixel = img.data + row * img.step;
		for (int col = 0; col < input_size_w; ++col) {
			data[3*i] = ((float)uc_pixel[0] - means[0])*stdvs[0];
			data[3*i+1] = ((float)uc_pixel[1] - means[1])*stdvs[0];
			data[3*i +2] = ((float)uc_pixel[2] - means[2])*stdvs[0];
			uc_pixel += 3;
			++i;
		}
	}
	//2. copy data to mlu.
	for (int i = 0; i < inputNum; i++) {
	  //拷贝数据MLU
      cnrtMemcpy(inputMluPtrS[i],
                 inputCpuPtrS[i],
                 inputSizeS[i],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);

    }
	
	//3. inference
	CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx_, param, cnrt_queue, nullptr));
    if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {

		// 4. get_data
		for (int i = 0; i < outputNum; i++) {
		  cnrtMemcpy(outputCpuPtrS[i],
					 outputMluPtrS[i],
					 outputSizeS[i],
					 CNRT_MEM_TRANS_DIR_DEV2HOST);
		}
		//float * result_landmarks = (reinterpret_cast<float*>(outputCpuPtrS[0]));
		float * result_angle = (reinterpret_cast<float*>(outputCpuPtrS[1]));
		/*for(int j = 0; j < 5; j++){
			cv::circle(img, cv::Point((result_landmarks[2*j]+0.5)*img.cols  , (result_landmarks[2*j+1]+0.5)*img.rows), 2, cv::Scalar(255,0,0), -1);
			std::cout<<"landmark:"<<result_landmarks[2*j]<<" "<<result_landmarks[2*j+1]<<std::endl;
		}
		std::cout<<"angle:"<<result_angle[0]<<" "<<result_angle[1]<<" "<<result_angle[2]<<std::endl;*/
		pose.yaw = result_angle[0]*90;
		pose.pitch = result_angle[1]*90;
		pose.roll = result_angle[2]*90;
		return pose;
    } else {
      std::cout << " SyncQueue Error ";
    }
}


HEADPose::~HEADPose(){

    for (int i = 0; i < inputNum; i++){
      cnrtFree(inputMluPtrS[i]);
	  
	  free(inputCpuPtrS[i]);
	}
	free(inputCpuPtrS);
	free(inputMluPtrS);
	
    for (int i = 0; i < outputNum; i++){
      cnrtFree(outputMluPtrS[i]);
	  
	  free(outputCpuPtrS[i]);
	  free(outTransCpuPtrS[i]);
	}
	free(outputCpuPtrS);
	free(outTransCpuPtrS);
	free(outputMluPtrS);
	free(param);
	
    cnrtDestroyQueue(cnrt_queue);
    cnrtDestroyFunction(function);
	
	cnrtUnloadModel(model);
  
	cnrtDestroy();
}