#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"


// * Each rect's data structure.
// * tlwh: topleft point & (w,h)
// * confidence: detection confidence.
// * feature: the rect's 128d feature.
// */
class DETECTION_ROW
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DETECTBOX tlwh;
    float confidence =1.0f;
	int classify = 0;
    FEATURE feature;
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
};

typedef std::vector<DETECTION_ROW, Eigen::aligned_allocator<DETECTION_ROW>> DETECTIONS;



#endif // MODEL_H
