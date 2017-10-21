
#include "sift_dog_pyramid.h"

#include "sift_exceptions.h"
#include "sift_ss_pyramid.h"

using namespace cv;
using namespace sift;


DogPyramid::DogPyramid()
	: ss ( nullptr )
{ }

DogPyramid::DogPyramid(const SsPyramid &ss) {
	start(ss);
}

void DogPyramid::start(const SsPyramid &ss) {
	check(ss);

	this->ss = &ss;

	step();
}

void DogPyramid::clear() {
	ss = nullptr;

	prev.release();
	curr.release();
	next.release();
}

void DogPyramid::step() {
	if (isEmpty())
		throw PyramidEmptyError("cannot step");

	if (ss->getInterval() == 0) {
		curr = ss->getCurrent() - ss->getPrevious();
	} else {
		prev = curr;
		curr = next;
		next.release();
	}

	next = ss->getNext() - ss->getCurrent();
}

void DogPyramid::check(const SsPyramid &ss) const {
	if (ss.isEmpty())
		throw PyramidEmptyError("need non-empty scale space pyramid");
}
