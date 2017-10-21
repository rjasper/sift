
#include "sift_ss_pyramid.h"

#include <stdexcept>

#include "sift_exceptions.h"

using namespace std;
using namespace cv;
using namespace sift;


SsPyramid::SsPyramid() { }

SsPyramid::SsPyramid(
	const Mat   &img,
		  bool  upscale,
		  int   intervals,
		  float sigma,
		  float sigmaInit)
{
	start(img, upscale, intervals, sigma, sigmaInit);
}

void SsPyramid::start(
	const Mat   &img,
		  bool  upscale,
		  int   intervals,
		  float sigma,
		  float sigmaInit)
{
	check(intervals, sigma, sigmaInit);

	this->intervals = intervals;
	this->sigma = sigma;

	init(img, upscale, sigmaInit);
}

void SsPyramid::clear() {
	if (isEmpty())
		return;

	prev.release();
	curr.release();
	next.release();
}

void SsPyramid::step() {
	if (isEmpty())
		throw PyramidEmptyError("cannot step");

	++interval;
	// if octave complete
	if (interval > intervals) {
		++octave;
		interval = 0;
		dsigma = sigma1;

		buildBase();
	// if octave incomplete
	} else {
		prev = curr;
		curr = next;
		next = Mat();
		dsigma *= k;

		buildNext();
	}
}

void SsPyramid::init(const Mat &img, bool upscale, float sigmaInit) {
	this->octave = upscale ? -1 : 0;
	this->interval = 0;

	this->k = pow(2.f, 1.f/intervals);

	float sigma0 = sigma * sqrt(k*k - 1.f);
	this->sigma1 = sigma0 * k;
	this->dsigma = sigma1;

	Mat img_init;
	float dsigma_init;

	// if upscaling then double height and width
	if (upscale) {
		dsigma_init = sqrt(sigma*sigma - 4.f*sigmaInit*sigmaInit);

		try {
			resize(img, img_init, Size(), OCTAVE_DOWNSAMPLING_DIVISOR, OCTAVE_DOWNSAMPLING_DIVISOR, CV_INTER_CUBIC);
		} catch (cv::Exception e) {
			throw BadImageFormat("cannot scale up", img);
		}
	} else {
		dsigma_init = sqrt(sigma*sigma - sigmaInit*sigmaInit);
		img_init = img;
	}

	// initial blur
	GaussianBlur(img_init, prev, Size(), dsigma_init);
	GaussianBlur(prev, curr, Size(), sigma0);
	GaussianBlur(curr, next, Size(), sigma1);
}

void SsPyramid::check(int intervals, float sigma, float sigmaInit) const {
	if (intervals < INTERVALS_MIN)
		throw domain_error("too few intervals");
	if (sigma <= 0.f)
		throw domain_error("sigma must be positive");
	if (sigmaInit < 0.f)
		throw domain_error("sigmaInit must be positive or zero");
}

void SsPyramid::buildBase() {
	downsample(prev);
	downsample(curr);
	downsample(next);
}

void SsPyramid::buildNext() {
	GaussianBlur(curr, next, Size(), dsigma);
}

void SsPyramid::downsample(Mat &mat) const {
	Size size_src = mat.size();
	Size size_dst(
		size_src.width  / OCTAVE_DOWNSAMPLING_DIVISOR,
		size_src.height / OCTAVE_DOWNSAMPLING_DIVISOR);

	if (size_dst.width == 0 || size_dst.height == 0)
		throw PyramidDownsamplingError("cannot downsample", octave);

	resize(mat, mat, size_dst, 0, 0, CV_INTER_NN);
}

