#include "retina_face.h"

//DEFINE_int32(mludevice, 0, "set using mlu device number, default: 0");

RetinaFace::RetinaFace(std::string models_path) {
	
	//FLAGS_alsologtostderr = 1;

	cnrtInit(0);
	unsigned devNum;
	cnrtGetDeviceCount(&devNum);
	if (devNum == 0)
	{
        std::cout<<"No device found"<<std::endl;
	}
	//获取指定设备的句柄
	cnrtGetDeviceHandle(&dev, dev_id);
	//设置当前使用的设备,作用于线程上下文
	cnrtSetCurrentDevice(dev);
	// 2. load model and get function

	//int size;
	
	//cnrtGetModelSize(models_path.c_str(), &size);
	
	cnrtLoadModel(&model, models_path.c_str());

	init();

	generate_anchor();
	std::cout<< "load retina model finished."<<std::endl;
}

void RetinaFace::init(){
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
	
}


void RetinaFace::generate_anchor() {
	myAnchors.clear();
	int width = input_size_w;
	int height = input_size_h;
	int featureMap[3][2];
	for (size_t i = 0; i < stride_list.size(); i++)
	{
		featureMap[i][0] = height / stride_list[i];
		featureMap[i][1] = width / stride_list[i];
	}

	for (size_t k = 0; k < 3; k++)
	{
		float ms = minSizes[2 * k];
		float ms2 = minSizes[2 * k + 1];
		int ftMapS1 = featureMap[k][0];
		int ftMapS2 = featureMap[k][1];
		for (size_t i = 0; i < ftMapS1; i++)
		{
			for (size_t j = 0; j < ftMapS2; j++)
			{
				float s_kx = ms / width;
				float s_ky = ms / height;
				float dense_cx = (j + 0.5f)*stride_list[k] / width;
				float dense_cy = (i + 0.5f)*stride_list[k] / height;

				float s_kx2 = ms2 / width;
				float s_ky2 = ms2 / height;

				RetinaAnchor myanchor;
				myanchor.anchor[0] = dense_cx;
				myanchor.anchor[1] = dense_cy;
				myanchor.anchor[2] = s_kx;
				myanchor.anchor[3] = s_ky;
				myAnchors.push_back(myanchor);

				RetinaAnchor myanchor2;
				myanchor2.anchor[0] = dense_cx;
				myanchor2.anchor[1] = dense_cy;
				myanchor2.anchor[2] = s_kx2;
				myanchor2.anchor[3] = s_ky2;
				myAnchors.push_back(myanchor2);
			}
		}
	}
}

static float variance[2] = { 0.1f, 0.2f };
static void bbox_pred(const float* anchor, const float* delta, cv::Rect2f& box) {
	float w = anchor[2];
	float h = anchor[3];

	float pred_ctr_x = delta[0] * variance[0] * w + anchor[0];
	float pred_ctr_y = delta[1] * variance[0] * h + anchor[1];
	float pred_w = std::exp(delta[2] * variance[1]) * w;
	float pred_h = std::exp(delta[3] * variance[1]) * h;

	box = cv::Rect2f(pred_ctr_x - 0.5f *pred_w,
		pred_ctr_y - 0.5f * pred_h,
		pred_w,
		pred_h);
}

static void landmk_pred(const float* anchor, const float* delta, float * landmark) {

	float w = anchor[2];
	float h = anchor[3];
	float x_ctr = anchor[0];
	float y_ctr = anchor[1];
	landmark[0] = x_ctr + delta[0] * variance[0] * w;
	landmark[1] = y_ctr + delta[1] * variance[0] * h;
	landmark[2] = x_ctr + delta[2] * variance[0] * w;
	landmark[3] = y_ctr + delta[3] * variance[0] * h;
	landmark[4] = x_ctr + delta[4] * variance[0] * w;
	landmark[5] = y_ctr + delta[5] * variance[0] * h;
	landmark[6] = x_ctr + delta[6] * variance[0] * w;
	landmark[7] = y_ctr + delta[7] * variance[0] * h;
	landmark[8] = x_ctr + delta[8] * variance[0] * w;
	landmark[9] = y_ctr + delta[9] * variance[0] * h;
}

static void nms_cpu(std::vector<FaceA>& boxes, float threshold, std::vector<FaceA>& filterOutBoxes) {
	filterOutBoxes.clear();
	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}
	//descending sort
	sort(boxes.begin(), boxes.end(), std::greater<FaceA>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx].rect.x, boxes[tmp_i].rect.x);
			float inter_y1 = std::max(boxes[good_idx].rect.y, boxes[tmp_i].rect.y);
			float inter_x2 = std::min(boxes[good_idx].rect.x + boxes[good_idx].rect.width, boxes[tmp_i].rect.x + boxes[tmp_i].rect.width);
			float inter_y2 = std::min(boxes[good_idx].rect.y + boxes[good_idx].rect.height, boxes[tmp_i].rect.y + boxes[tmp_i].rect.height);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx].rect.width + 1) * (boxes[good_idx].rect.height + 1);
			float area_2 = (boxes[tmp_i].rect.width + 1) * (boxes[tmp_i].rect.height + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

static float get_face_radio(float * landmarks, cv::Rect frect) {
	float width = frect.width;
	float d_eye2 = (landmarks[0] - landmarks[2])*(landmarks[0] - landmarks[2]) + (landmarks[1] - landmarks[3])*(landmarks[1] - landmarks[3]);
	float e_c_x = (landmarks[0] + landmarks[2]) / 2;
	float e_c_y = (landmarks[1] + landmarks[3]) / 2;
	float d_nose_eye = sqrt((landmarks[4] - e_c_x)*(landmarks[4] - e_c_x) + (landmarks[5] - e_c_y)*(landmarks[5] - e_c_y));
	if (d_eye2 == 0) {
		return 0.0f;
	}
	return d_eye2 / (d_nose_eye*width);
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


void RetinaFace::sure_face(std::vector<FaceA>&faces){
	for (int i = 0; i < faces.size();i++)
	{
		float fradio = get_face_radio(faces[i].landmarks, faces[i].rect);
		if (fradio > fradio_threadhold && faces[i].rect.width > min_width) {
			return;
		}else{
			faces[i].score = default_score;
		}
	}
}

static void PrintTime(double start, std::string Tag)
{
	double end = cv::getTickCount();
	std::cout<<Tag << (end-start)/cv::getTickFrequency()*1000<<" ms"<<std::endl;
}

void RetinaFace::Detect(cv::Mat& img, std::vector<FaceA>&face) {
	
	//double start = cv::getTickCount();
	
	cv::Mat dst;
	int pad_left = 0, resize_w = 0, pad_top = 0, resize_h = 0;
	dst = resize_with_crop(img, input_size_w, input_size_h, pad_top, pad_left, resize_w, resize_h);
	
	//PrintTime(start, "resize_with_crop:");
	//start = cv::getTickCount();
	//1. process
	float * data = reinterpret_cast<float*>(inputCpuPtrS[0]);
	
	int i = 0;
	for (int row = 0; row < input_size_h; ++row) {
		uchar* uc_pixel = dst.data + row * dst.step;
		for (int col = 0; col < input_size_w; ++col) {
			data[0] = (float)uc_pixel[0] - means[0];
			data[1] = (float)uc_pixel[1] - means[1];
			data[2] = (float)uc_pixel[2] - means[2];
			uc_pixel += 3;
			data += 3;
			++i;
		}
	}
	//PrintTime(start, "process:");
	//start = cv::getTickCount();
	
	//2. copy data to mlu.
	for (int i = 0; i < inputNum; i++) {
	  //拷贝数据MLU
      cnrtMemcpy(inputMluPtrS[i],
                 inputCpuPtrS[i],
                 inputSizeS[i],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);

    }
	//PrintTime(start, "CNRT_MEM_TRANS_DIR_HOST2DEV:");
	//start = cv::getTickCount();
	
	//3. inference
	CNRT_CHECK(cnrtInvokeRuntimeContext(rt_ctx_, param, cnrt_queue, &invokeParam));
    if (cnrtSyncQueue(cnrt_queue) == CNRT_RET_SUCCESS) {
		
		//PrintTime(start, "cnrtInvokeRuntimeContext:");
		//start = cv::getTickCount();
		// 4. get_data
		for (int i = 0; i < outputNum; i++) {
		  int output_count = outputSizeS[i] / cnrtDataTypeSize(output_data_type[i]);
		  cnrtMemcpy(outTransCpuPtrS[i],
					 outputMluPtrS[i],
					 outputSizeS[i],
					 CNRT_MEM_TRANS_DIR_DEV2HOST);
		  /*std::vector<int> shape(4, 1);
		  int dimNum = 4;
		  cnrtGetOutputDataShape((int**)&shape, &dimNum, i, function);
		  int dim_order[4] = {0, 3, 1, 2};
		  int dim_shape[4] = {shape[0], shape[1],
							  shape[2], shape[3]};  // NHWC
		  cnrtTransDataOrder(outputCpuPtrS[i], CNRT_FLOAT32, outTransCpuPtrS[i],
							 4, dim_shape, dim_order);*/
		}
		
		//PrintTime(start, "CNRT_MEM_TRANS_DIR_DEV2HOST:");
		//start = cv::getTickCount();
		
		float * result_loc = (reinterpret_cast<float*>(outTransCpuPtrS[0]));
		float * result_conf = (reinterpret_cast<float*>(outTransCpuPtrS[1]));
		float * result_landmks = (reinterpret_cast<float*>(outTransCpuPtrS[2]));

		float w_scale = 0.0f, h_scale = 0.0f;
		if (pad_left != 0) {
			w_scale = float(img.cols) / resize_w;
			h_scale = float(img.rows) / input_size_h;
		}
		else {
			w_scale = float(img.cols) / input_size_w;
			h_scale = float(img.rows) / resize_h;
		}
		
		std::vector<FaceA>pre_plates;
		int length = out_count[1] / 2;
		int length_array[10] = {0, length, 2*length, 3*length,4*length, 5*length, 6*length, 7*length, 8*length, 9*length}; 
		//std::cout<<"out_count:"<<out_count[1]<<std::endl;
		
		float temp_loc_data[4];
		float temp_landmark_data[10];
		for (int i = 0; i < length; i++)
		{
			float score = result_conf[length + i];
			if (score > threhold)
			{
				FaceA ps;
				ps.score = score;
				
				for (int j = 0; j < 4; j++){
					temp_loc_data[j] = result_loc[length_array[j]+i];
				}
				bbox_pred(myAnchors[i].anchor, temp_loc_data, ps.rect);
				ps.rect.x = std::max((ps.rect.x*input_size_w - pad_left)*w_scale, 0.0f);
				ps.rect.y = std::max((ps.rect.y*input_size_h - pad_top)*h_scale, 0.0f);
				ps.rect.width = std::min(ps.rect.width*input_size_w * w_scale, img.cols - ps.rect.x);
				ps.rect.height = std::min(ps.rect.height*input_size_h * h_scale, img.rows - ps.rect.y);
				if (ps.rect.width < 10 || ps.rect.height < 10) {
					continue;
				}
				
				for (int j = 0; j < 10; j++){
					temp_landmark_data[j] = result_landmks[length_array[j]+i];
				}
				landmk_pred(myAnchors[i].anchor, temp_landmark_data, ps.landmarks);
				for (size_t n = 0; n < 5; n++)
				{
					ps.landmarks[2 * n] = (ps.landmarks[2 * n] * input_size_w - pad_left) * w_scale;
					ps.landmarks[2 * n + 1] = (ps.landmarks[2 * n + 1] * input_size_h - pad_top) * h_scale;
				}
				pre_plates.push_back(ps);
			}
		}
		nms_cpu(pre_plates, iou_threhold, face);
		sure_face(face);
		//PrintTime(start, "sure_face:");
    } else {
      std::cout<< " SyncQueue Error "<<std::endl;
    }
}


RetinaFace::~RetinaFace(){

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