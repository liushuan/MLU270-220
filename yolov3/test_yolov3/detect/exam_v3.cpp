#include "exam_v3.h"

//DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");

EXAMDetect::EXAMDetect(std::string models_path) {
	//FLAGS_alsologtostderr = 1;
	cnrtInit(0);
	//unsigned devNum;
	//cnrtGetDeviceCount(&devNum);
	/*if (FLAGS_mludevice >= 0) {
		CHECK_NE(devNum, 0) << "No device found";
		CHECK_LE(FLAGS_mludevice, devNum) << "valid device count: " << devNum;
	} else {
		std::cout << "Invalid device number"<<std::endl;
	}
	cnrtGetDeviceHandle(&dev, FLAGS_mludevice);*/
	
	//cnrtSetCurrentDevice(dev);
	// 2. load model and get function

	//int size;
	
	//cnrtGetModelSize(models_path.c_str(), &size);
	cnrtLoadModel(&model, models_path.c_str());
	init();
	std::cout<< "load retina model finished."<<std::endl;
}

void EXAMDetect::init(){
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

static cv::Mat resize_with_crop(cv::Mat origin_image, int input_image_size_w, int input_image_size_h, int &pad_top, int &pad_left, int &resize_w, int &resize_h) {

	float radio = (float)(input_image_size_w) / input_image_size_h;
	cv::Mat  resized_image;
	int h = origin_image.rows;
	int w = origin_image.cols;
	//int pad_left = 0, resize_w = 0, pad_top = 0, resize_h = 0;
	if (h*radio > w) {
		resize_w = int((float)w / h * input_image_size_h);
		cv::resize(origin_image, resized_image, cv::Size(resize_w, input_image_size_h));
		pad_left = (input_image_size_w - resize_w) / 2;
		int pad_wright = (input_image_size_w - resize_w + 1) / 2;
		cv::copyMakeBorder(resized_image, resized_image, 0, 0, pad_left, pad_wright, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	else {
		resize_h = int((float)h / w * input_image_size_w);
		cv::resize(origin_image, resized_image, cv::Size(input_image_size_w, resize_h));
		pad_top = (input_image_size_h - resize_h) / 2;
		int pad_boom = (input_image_size_h - resize_h + 1) / 2;
		cv::copyMakeBorder(resized_image, resized_image, pad_top, pad_boom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	return resized_image;
}


void EXAMDetect::Detect(cv::Mat& img, std::vector<Box>&exams) {
	cv::Mat dst;
	int pad_left = 0, resize_w = 0, pad_top = 0, resize_h = 0;
	dst = resize_with_crop(img, input_size_w, input_size_h, pad_top, pad_left, resize_w, resize_h);
	//1. process
	cv::imwrite("output/dst.jpg", dst);
	float * data = reinterpret_cast<float*>(inputCpuPtrS[0]);
	int i = 0;
	for (int row = 0; row < input_size_h; ++row) {
		uchar* uc_pixel = dst.data + row * dst.step;
		for (int col = 0; col < input_size_w; ++col) {
			data[3*i] = (float)uc_pixel[2] * stdvs[2];
			data[3*i+1] = (float)uc_pixel[1] * stdvs[1];
			data[3*i +2] = (float)uc_pixel[0] * stdvs[0];
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
			//std::cout<<"data:"<<outputSizeS[i]<<std::endl;
			
			if (output_data_type[i] != CNRT_FLOAT32) {
				//int output_count = infr->outCount()[i];
				int output_count =outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
				//std::cout<<"output_count:"<<output_count<<std::endl;
				cnrtCastDataType(outputCpuPtrS[i],
                         output_data_type[i],
                         outTransCpuPtrS[i],
                         CNRT_FLOAT32,
                         output_count,
                         nullptr);
			} else {
				memcpy(outTransCpuPtrS[i], outputCpuPtrS[i], outputSizeS[i]);
			}
			
		}
		
		float * result_data = (reinterpret_cast<float*>(outTransCpuPtrS[0]));
		
		float w_scale = 0.0f, h_scale = 0.0f;
		if (pad_left != 0) {
			w_scale = float(img.cols) / resize_w;
			h_scale = float(img.rows) / input_size_h;
		}
		else {
			w_scale = float(img.cols) / input_size_w;
			h_scale = float(img.rows) / resize_h;
		}
		int box_size = result_data[0];
		for (int i = 0; i < box_size; i++)
		{
			float batch_index = result_data[64+7*i];
			float class_indx = result_data[64+7*i + 1];
			float score = result_data[64+7*i + 2];
			Box box;
			box.classes = result_data[64+7*i + 1];
			box.score = result_data[64+7*i + 2];
			box.xmin = (result_data[64+7*i + 3] * input_size_w - pad_left)*w_scale;
			box.ymin = (result_data[64+7*i + 4] * input_size_h - pad_top) *h_scale;
			box.xmax = (result_data[64+7*i + 5] * input_size_w - pad_left)*w_scale;
			box.ymax = (result_data[64+7*i + 6] * input_size_h - pad_top) *h_scale;
			//std::cout<<"value:"<<batch_index<<" "<<box.classes<<" "<<box.score<<" "<<box.xmin<<" "<<box.ymin<<" "<<box.xmax<<" "<<box.ymax<<std::endl;
			//ps.rect.x = std::max((ps.rect.x*input_size_w - pad_left)*w_scale, 0.0f);
			//ps.rect.y = std::max((ps.rect.y*input_size_h - pad_top)*h_scale, 0.0f);
			if (box.xmax - box.xmin < 10 || box.ymax - box.ymin < 10){
				continue;
			}else{
				exams.push_back(box);
			}
		}
		
    } else {
      std::cout<< " SyncQueue Error "<<std::endl;
    }
}

EXAMDetect::~EXAMDetect(){

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