
#include "sift_exceptions.h"

using namespace std;
using namespace cv;
using namespace sift;


PyramidEmptyError::PyramidEmptyError(const string &msg)
	: logic_error ( msg )
{ }

PyramidDownsamplingError::PyramidDownsamplingError(const std::string &msg, int octave)
	: logic_error ( msg    )
	, octave      ( octave )
{ }


BadImageFormat::BadImageFormat(const std::string &msg, const Mat &img)
	: invalid_argument ( msg )
	, img              ( img )
{ }
