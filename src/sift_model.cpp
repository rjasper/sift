
#include "sift_model.h"

#include <utility>
#include <stdexcept>

#include <list>

#include "sift.h"
#include "sift_io.h"
#include "sift_exceptions.h"
#include "sift_detector.h"

using namespace std;
using namespace cv;
using namespace sift;


Model::Model() {
	init();
}

Model::Model(const Model &other) {
	init();

	*this = other;
}

Model::Model(Model &&other) {
	*this = move(other);
}

Model::Model(
	const vector<KeyPoint> &keyPoints,
	const Mat              &descriptors)
	: keyPoints   ( keyPoints   )
	, descriptors ( descriptors )
{
	checkFeatures(keyPoints, descriptors);

	init();
}

Model::Model(
	const vector<KeyPoint> &&keyPoints,
	const Mat              &descriptors)
	: keyPoints   ( move(keyPoints) )
	, descriptors ( descriptors     )
{
	checkFeatures(keyPoints, descriptors);

	init();
}

Model::~Model() {
	delete matcher;
}

Model &Model::operator=(const Model &rhs) {
	if (&rhs == this)
		return *this;

	if (matcher) {
		delete matcher;
		matcher = nullptr;
	}

	keyPoints   = rhs.keyPoints;
	descriptors = rhs.descriptors;
	
	return *this;
}

Model &Model::operator=(Model &&rhs) {
	if (&rhs == this)
		return *this;

	keyPoints = move(rhs.keyPoints);

	descriptors = rhs.descriptors; // doesn't support move yet
	rhs.descriptors.release();

	matcher = rhs.matcher;
	rhs.matcher = nullptr;

	return *this;
}

void Model::setFeatures(
	const vector<KeyPoint> &keyPoints,
	const Mat              &descriptors)
{
	checkFeatures(keyPoints, descriptors);

	this->keyPoints   = keyPoints;
	this->descriptors = descriptors; // no data copy
}

void Model::setFeatures(
	const vector<KeyPoint> &&keyPoints,
	const Mat              &descriptors)
{
	checkFeatures(keyPoints, descriptors);

	this->keyPoints   = move(keyPoints);
	this->descriptors = descriptors; // no data copy
}

int Model::detect(const Detector &detector, const Mat &img) {
	clear();

	return detector.detect(img, keyPoints, descriptors);
}

int Model::match(const Mat &descriptors, vector<DMatch> &matches) const {
	if (descriptors.cols != this->descriptors.cols)
		throw invalid_argument("descriptor dimension doesn't match");
	if (descriptors.type() != CV_32F)
		throw invalid_argument("descriptors must be of type CV_32F");

	return matchImpl(descriptors, matches);
}

int Model::match(const Model &model, vector<DMatch> &matches) const {
	return matchImpl(model.descriptors, matches);
}

void Model::filter(const function<bool (const KeyPoint &)> &condition) {
	list<int> indices;
	int n = keyPoints.size();

	// collect filtered indices
	for (int i = 0; i < n; ++i) {
		const KeyPoint &kp = keyPoints[i];

		if (condition(kp))
			indices.push_back(i);
	}
	
	// apply filtering

	// new size
	int m = indices.size();

	list<int>::iterator
		&begin = indices.begin(),
		&end   = indices.end(),
		idx;
	// move accepted elements to the left
	// for each filtered key point
	int i;
	for (idx = begin, i = 0; idx != end; ++idx, ++i) {
		keyPoints[i] = keyPoints[*idx];

		Mat &src = descriptors.row(*idx);
		Mat &dst = descriptors.row(i);
		src.copyTo(dst);
	}

	// note that the capacities may not be shrinked
	keyPoints.resize(m);
	descriptors.resize(m);
}

void Model::map(const function<void (KeyPoint &)> &mapper) {
	vector<KeyPoint>::iterator
		&begin = keyPoints.begin(),
		&end   = keyPoints.end(),
		kp;
	// for each key point
	for (kp = begin; kp != end; ++kp)
		mapper(*kp);
}

void Model::clear() {
	keyPoints.clear();
	descriptors.release();

	init();
}

void Model::init() {
	matcher = nullptr;
}

void Model::train() const {
	matcher = new FlannBasedMatcher();
	vector<Mat> v(1, descriptors);
	matcher->add(v);
}

int Model::matchImpl(const Mat &descriptors, vector<DMatch> &matches) const {
	if (!matcher)
		train();
	
	// the k nearest neighbours where k := K_NEAREST_NEIGHBORS
	vector<vector<DMatch>> knn;
	matcher->knnMatch(descriptors, knn, K_NEAREST_NEIGHBORS);

	for each (const vector<DMatch> nbrs in knn) {
		if (nbrs.size() < K_NEAREST_NEIGHBORS)
			continue;

		float d0 = nbrs[0].distance;
		float d1 = nbrs[1].distance;

		if (d0 / d1 <= MATCH_NN_DISTANCE_RATIO)
			matches.push_back(nbrs[0]); // TODO avoid pushing
	}

	return matches.size();
}

void Model::checkFeatures(
	const vector<KeyPoint> &keyPoints,
	const Mat              &descriptors) const
{
	if (keyPoints.size() != descriptors.rows)
		throw invalid_argument("key point and descriptor amount must be equal");
	if (descriptors.cols <= 0)
		throw invalid_argument("descriptor dimension must be positive");
	if (descriptors.type() != CV_32F)
		throw invalid_argument("descriptor must be of type CV_32F");
}

void Model::read(istream &in) {
	readFeatures(in, keyPoints, descriptors);
}

void Model::write(ostream &out) const {
	writeFeatures(out, keyPoints, descriptors);
}
