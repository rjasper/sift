
#include "sift_io.h"

using namespace std;
using namespace cv;


static const int DESCR_ENTRIES_PER_LINE = 16;


namespace sift {


void writeFeatures(
	      ostream          &out,
	const vector<KeyPoint> &keyPoints,
	const Mat              &descriptors)
{
	int n = keyPoints.size();
	int dim = descriptors.cols;

	out << n <<" "<< dim << endl;

	for (int i = 0; i < n; ++i) {
		const KeyPoint &k = keyPoints[i];

		out << k.pt.y <<" "<< k.pt.x <<" "<< k.size <<" "<< k.angle << endl;

		for (int j = 0; j < dim; ++j) {
			out << cvFloor(descriptors.at<float>(i, j));

			// if last entry in line
			if (j % DESCR_ENTRIES_PER_LINE == DESCR_ENTRIES_PER_LINE - 1 || j == dim - 1)
				out << endl;
			else
				out <<" ";
		}
	}
}

void readFeatures(
	istream          &in,
	vector<KeyPoint> &keyPoints,
	Mat              &descriptors)
{
	int n, dim;

	in >> n >> dim;
	keyPoints.resize(n);
	descriptors = Mat(n, dim, CV_32F);

	for (int i = 0; i < n; ++i) {
		KeyPoint &k = keyPoints[i];

		in >> k.pt.y >> k.pt.x >> k.size >> k.angle;

		for (int j = 0; j < dim; ++j)
			in >> descriptors.at<float>(i, j);
	}
}


}
