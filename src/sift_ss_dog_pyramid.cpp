
#include "sift_ss_dog_pyramid.h"

#include "sift.h"

using namespace cv;
using namespace sift;


SsDogPyramid::SsDogPyramid() { }

SsDogPyramid::SsDogPyramid(
	const Mat   &img,
		  bool  upscale,
		  int   intervals,
		  float sigma,
		  float sigmaInit) 
	: ss  ( img, upscale, intervals, sigma, sigmaInit )
	, dog ( ss                                        )
{
	init();
}

void SsDogPyramid::init() {
	// build dog base
	ss.step();
	dog.step();
}

Size SsDogPyramid::getCurrentSize() const {
	return getCurrentDog().size();
}

void SsDogPyramid::start(
	const Mat   &img,
		  bool  upscale,
		  int   intervals,
		  float sigma,
		  float sigmaInit)
{
	ss.start(img, upscale, intervals, sigma, sigmaInit);
	dog.start(ss);

	init();
}

void SsDogPyramid::clear() {
	ss.clear();
	dog.clear();
}

void SsDogPyramid::step() {
	ss.step();
	dog.step();

	// if octave was completed
	if (ss.getInterval() == 0) {
		// build dog base
		ss.step();
		dog.step();
	}
}
