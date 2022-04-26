#include "exam_v3.h"


static void init(ExamModel & exam_model, std::string models_path)
{
	
	//获取指定设备的句柄
	cnrtGetDeviceHandle(&exam_model.dev, exam_model.dev_id);
	//设置当前使用的设备,作用于线程上下文
	cnrtSetCurrentDevice(exam_model.dev);
	cnrtLoadModel(&exam_model.model, models_path.c_str());
	
	cnrtCreateFunction(&exam_model.function);
    cnrtExtractFunction(&exam_model.function, exam_model.model, "subnet0");
	if(exam_model.dev_channel>=0)
	{
		CNRT_CHECK(cnrtSetCurrentChannel((cnrtChannelType_t)exam_model.dev_channel));
	}
	
    cnrtCreateRuntimeContext(&exam_model.rt_ctx_, exam_model.function, NULL);
	
    // 3. get function's I/O DataDesc
    cnrtGetInputDataSize(&exam_model.inputSizeS, &exam_model.inputNum, exam_model.function);
    cnrtGetOutputDataSize(&exam_model.outputSizeS, &exam_model.outputNum, exam_model.function);
    cnrtGetInputDataType(&exam_model.input_data_type, &exam_model.inputNum, exam_model.function);
    cnrtGetOutputDataType(&exam_model.output_data_type, &exam_model.outputNum, exam_model.function);
	
    // 4. allocate I/O data space on CPU memory and prepare Input data
	exam_model.inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * exam_model.inputNum));
    exam_model.outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * exam_model.outputNum));
	exam_model.outTransCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * exam_model.outputNum));
    exam_model.param = reinterpret_cast<void**>(malloc(sizeof(void*) * (exam_model.inputNum + exam_model.outputNum)));

    for (int i = 0; i < exam_model.inputNum; i++) {
      int ip = exam_model.inputSizeS[i] / cnrtDataTypeSize(exam_model.input_data_type[i]);
      auto databuf = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
	  //get_input_data("imgs/test2.jpg", databuf);
      exam_model.in_count.push_back(ip);
      exam_model.inputCpuPtrS[i] = reinterpret_cast<void*>(databuf);  // NHWC
    }

    for (int i = 0; i < exam_model.outputNum; i++) {
      int op = exam_model.outputSizeS[i] / cnrtDataTypeSize(exam_model.output_data_type[i]);
      float* outcpu = reinterpret_cast<float*>(malloc(op * sizeof(float)));
      exam_model.out_count.push_back(op);
      exam_model.outputCpuPtrS[i] = reinterpret_cast<void*>(outcpu);
	  
	  float* outcpu1 = reinterpret_cast<float*>(malloc(op * sizeof(float)));
	  exam_model.outTransCpuPtrS[i] = reinterpret_cast<void*>(outcpu1);
    }
	// 5. allocate I/O data space on MLU memory and copy Input data
    // Only 1 batch so far
    exam_model.inputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * exam_model.inputNum));
    exam_model.outputMluPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * exam_model.outputNum));
    for (int i = 0; i < exam_model.inputNum; i++) {
      cnrtMalloc(&exam_model.inputMluPtrS[i], exam_model.inputSizeS[i]);
    }
    for (int i = 0; i < exam_model.outputNum; i++) {
      cnrtMalloc(&exam_model.outputMluPtrS[i], exam_model.outputSizeS[i]);
    }
    for (int i = 0; i < exam_model.inputNum; i++) {
      exam_model.param[i] = exam_model.inputMluPtrS[i];
    }
    for (int i = 0; i < exam_model.outputNum; i++) {
      exam_model.param[exam_model.inputNum + i] = exam_model.outputMluPtrS[i];
    }
	 // 6. create cnrt_queue
    cnrtCreateQueue(&exam_model.cnrt_queue);
    //cnrtSetRuntimeContextDeviceId(rt_ctx_, dev);  //Dev ordinal value error
	cnrtInitRuntimeContext(exam_model.rt_ctx_, NULL);
	
	
	//设置invoke的参数
	unsigned int affinity=1<<exam_model.dev_channel;//设置通道亲和性,使用指定的MLU cluster做推理
	exam_model.invokeParam.invoke_param_type = CNRT_INVOKE_PARAM_TYPE_0;
	exam_model.invokeParam.cluster_affinity.affinity = &affinity;
	
}


static void release(ExamModel & exam_model){
	for (int i = 0; i < exam_model.inputNum; i++){
      cnrtFree(exam_model.inputMluPtrS[i]);
	  
	  free(exam_model.inputCpuPtrS[i]);
	}
	free(exam_model.inputCpuPtrS);
	free(exam_model.inputMluPtrS);
	
    for (int i = 0; i < exam_model.outputNum; i++){
      cnrtFree(exam_model.outputMluPtrS[i]);
	  
	  free(exam_model.outputCpuPtrS[i]);
	  free(exam_model.outTransCpuPtrS[i]);
	}
	free(exam_model.outputCpuPtrS);
	free(exam_model.outTransCpuPtrS);
	free(exam_model.outputMluPtrS);
	free(exam_model.param);
	
    cnrtDestroyQueue(exam_model.cnrt_queue);
    cnrtDestroyFunction(exam_model.function);
	
	cnrtUnloadModel(exam_model.model);
}

EXAMDetect::EXAMDetect(std::string models_path, std::string models_batch_path) {

	cnrtInit(0);
	unsigned devNum;
	cnrtGetDeviceCount(&devNum);
	if (devNum == 0)
	{
        std::cout<<"No device found"<<std::endl;
	}
	
	init(exam_model, models_path);
	init(exam_model_batch, models_batch_path);
	std::cout<< "load model finished."<<std::endl;

}

/*void EXAMDetect::init(){
	cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, "subnet0");
	if(dev_channel>=0)
	{
		CNRT_CHECK(cnrtSetCurrentChannel((cnrtChannelType_t)dev_channel));
	}
	
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
	
	
	//设置invoke的参数
	unsigned int affinity=1<<dev_channel;//设置通道亲和性,使用指定的MLU cluster做推理
	invokeParam.invoke_param_type = CNRT_INVOKE_PARAM_TYPE_0;
	invokeParam.cluster_affinity.affinity = &affinity;
	
}*/

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
	float * data = reinterpret_cast<float*>(exam_model.inputCpuPtrS[0]);
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
	for (int i = 0; i < exam_model.inputNum; i++) {
	  //拷贝数据MLU
      cnrtMemcpy(exam_model.inputMluPtrS[i],
                 exam_model.inputCpuPtrS[i],
                 exam_model.inputSizeS[i],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);

    }
	
	//3. inference
	CNRT_CHECK(cnrtInvokeRuntimeContext(exam_model.rt_ctx_, exam_model.param, exam_model.cnrt_queue, &exam_model.invokeParam));
    if (cnrtSyncQueue(exam_model.cnrt_queue) == CNRT_RET_SUCCESS) {

			// 4. get_data
		for (int i = 0; i < exam_model.outputNum; i++) {
		    cnrtMemcpy(exam_model.outputCpuPtrS[i],
					 exam_model.outputMluPtrS[i],
					 exam_model.outputSizeS[i],
					 CNRT_MEM_TRANS_DIR_DEV2HOST);
			//std::cout<<"data:"<<outputSizeS[i]<<std::endl;
			
			if (exam_model.output_data_type[i] != CNRT_FLOAT32) {
				//int output_count = infr->outCount()[i];
				int output_count =exam_model.outputSizeS[i] / cnrtDataTypeSize(exam_model.output_data_type[i]);
				//std::cout<<"output_count:"<<output_count<<std::endl;
				cnrtCastDataType(exam_model.outputCpuPtrS[i],
                         exam_model.output_data_type[i],
                         exam_model.outTransCpuPtrS[i],
                         CNRT_FLOAT32,
                         output_count,
                         nullptr);
			} else {
				memcpy(exam_model.outTransCpuPtrS[i], exam_model.outputCpuPtrS[i], exam_model.outputSizeS[i]);
			}
			
		}
		
		float * result_data = (reinterpret_cast<float*>(exam_model.outTransCpuPtrS[0]));
		
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
			}else if (box.score > threhold){
				exams.push_back(box);
			}
		}
		
    } else {
      std::cout<< " SyncQueue Error "<<std::endl;
    }
}


void EXAMDetect::Detect(std::vector<cv::Mat>&imgs, std::vector<std::vector<Box>>& exams){
	int real_batch_size = imgs.size() >= batch_size ? batch_size:imgs.size();
	exams.resize(real_batch_size);
	std::vector<int>pad_lefts, pad_tops, resize_ws, resize_hs;
	float * data = reinterpret_cast<float*>(exam_model_batch.inputCpuPtrS[0]);
	
	for(int k = 0; k < real_batch_size; k++){
		cv::Mat dst;
		int pad_left = 0, resize_w = 0, pad_top = 0, resize_h = 0;
		dst = resize_with_crop(imgs[k], input_size_w, input_size_h, pad_top, pad_left, resize_w, resize_h);
		//1. process
		pad_lefts.push_back(pad_left);
		pad_tops.push_back(pad_top);
		resize_ws.push_back(resize_w);
		resize_hs.push_back(resize_h);

		for (int row = 0; row < input_size_h; ++row) {
			uchar* uc_pixel = dst.data + row * dst.step;
			for (int col = 0; col < input_size_w; ++col) {
				data[0] = (float)uc_pixel[2] * stdvs[2];
				data[1] = (float)uc_pixel[1] * stdvs[1];
				data[2] = (float)uc_pixel[0] * stdvs[0];
				uc_pixel += 3;
				data += 3;
			}
		}
	}
	
	//2. copy data to mlu.
	for (int i = 0; i < exam_model_batch.inputNum; i++) {
	  //拷贝数据MLU
      cnrtMemcpy(exam_model_batch.inputMluPtrS[i],
                 exam_model_batch.inputCpuPtrS[i],
                 exam_model_batch.inputSizeS[i],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);

    }
	
	//3. inference
	CNRT_CHECK(cnrtInvokeRuntimeContext(exam_model_batch.rt_ctx_, exam_model_batch.param, exam_model_batch.cnrt_queue, &exam_model_batch.invokeParam));
    if (cnrtSyncQueue(exam_model_batch.cnrt_queue) == CNRT_RET_SUCCESS) {

			// 4. get_data
		for (int i = 0; i < exam_model_batch.outputNum; i++) {
		    cnrtMemcpy(exam_model_batch.outputCpuPtrS[i],
					 exam_model_batch.outputMluPtrS[i],
					 exam_model_batch.outputSizeS[i],
					 CNRT_MEM_TRANS_DIR_DEV2HOST);
			//std::cout<<"data:"<<outputSizeS[i]<<std::endl;
			
			if (exam_model_batch.output_data_type[i] != CNRT_FLOAT32) {
				//int output_count = infr->outCount()[i];
				int output_count =exam_model_batch.outputSizeS[i] / cnrtDataTypeSize(exam_model_batch.output_data_type[i]);
				//std::cout<<"output_count:"<<output_count<<std::endl;
				cnrtCastDataType(exam_model_batch.outputCpuPtrS[i],
                         exam_model_batch.output_data_type[i],
                         exam_model_batch.outTransCpuPtrS[i],
                         CNRT_FLOAT32,
                         output_count,
                         nullptr);
			} else {
				memcpy(exam_model_batch.outTransCpuPtrS[i], exam_model_batch.outputCpuPtrS[i], exam_model_batch.outputSizeS[i]);
			}
			
		}
		
		float * result_data = (reinterpret_cast<float*>(exam_model_batch.outTransCpuPtrS[0]));
		int sBatchsize =exam_model_batch.outputSizeS[0] / cnrtDataTypeSize(exam_model_batch.output_data_type[0]) / batch_size;
		for(int k = 0; k < real_batch_size; k++)
		{
			float w_scale = 0.0f, h_scale = 0.0f;
			if (pad_lefts[k] != 0) {
				w_scale = float(imgs[k].cols) / resize_ws[k];
				h_scale = float(imgs[k].rows) / input_size_h;
			}
			else {
				w_scale = float(imgs[k].cols) / input_size_w;
				h_scale = float(imgs[k].rows) / resize_hs[k];
			}
			
			int box_size = result_data[k*sBatchsize];
			for (int i = 0; i < box_size; i++)
			{
				float batch_index = result_data[k*sBatchsize + 64+7*i];
				if ((batch_index < 0) || (batch_index >= real_batch_size)) {
					continue;
				}

				float class_indx = result_data[k*sBatchsize + 64+7*i + 1];
				float score = result_data[k*sBatchsize + 64+7*i + 2];
				Box box;
				box.classes = result_data[k*sBatchsize + 64+7*i + 1];
				box.score = result_data[k*sBatchsize + 64+7*i + 2];
				box.xmin = (result_data[k*sBatchsize + 64+7*i + 3] * input_size_w - pad_lefts[k])*w_scale;
				box.ymin = (result_data[k*sBatchsize + 64+7*i + 4] * input_size_h - pad_tops[k]) *h_scale;
				box.xmax = (result_data[k*sBatchsize + 64+7*i + 5] * input_size_w - pad_lefts[k])*w_scale;
				box.ymax = (result_data[k*sBatchsize + 64+7*i + 6] * input_size_h - pad_tops[k]) *h_scale;
				//std::cout<<"value:"<<batch_index<<" "<<box.classes<<" "<<box.score<<" "<<box.xmin<<" "<<box.ymin<<" "<<box.xmax<<" "<<box.ymax<<std::endl;
				//ps.rect.x = std::max((ps.rect.x*input_size_w - pad_left)*w_scale, 0.0f);
				//ps.rect.y = std::max((ps.rect.y*input_size_h - pad_top)*h_scale, 0.0f);
				if (box.xmax - box.xmin < 10 || box.ymax - box.ymin < 10){
					continue;
				}else if (box.score > threhold){
					exams[batch_index].push_back(box);
				}
			}
		}
		
    } else {
      std::cout<< " SyncQueue Error "<<std::endl;
    }
}




EXAMDetect::~EXAMDetect(){

	release(exam_model);
	release(exam_model_batch);
	
	cnrtDestroy();
}