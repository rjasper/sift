
#include "sift_detector.h"

#include <stdint.h>
#include <stdexcept>
#include <memory>
#include <math.h>

#include "sift_exceptions.h"
#include "sift_ss_dog_pyramid.h"
#include "sift_candidate.h"
#include "sift_model.h"

using namespace std;
using namespace cv;
using namespace sift;


static const float PI_F = 3.1415927f;


Detector::Detector()
	// default settings
	: upscale                      ( UPSCALE                        )
	, borderWidth                  ( IMAGE_BORDER                   )
	, intervals                    ( INTERVALS                      )
	, sigma                        ( SIGMA                          )
	, sigmaInit                    ( SIGMA_INIT                     )
	, contrastThreshold            ( CONTRAST_THRESHOLD             )
	, curvatureThreshold           ( CURVATURE_THRESHOLD            )
	, orientationSigmaFactor       ( ORIENTATION_SIGMA_FACTOR       )
	, orientationRadiusFactor      ( ORIENTATION_RADIUS_FACTOR      )
	, orientationPeakRatio         ( ORIENTATION_PEAK_RATIO         )
	, orientationSmoothIterations  ( ORIENTATION_SMOOTH_ITERATIONS  )
	, descriptorSigmaFactor        ( DESCRIPTOR_SIGMA_FACTOR        )
	, descriptorMagnitudeThreshold ( DESCRIPTOR_MAGNITUDE_THRESHOLD )
	// initial state
	, descriptorVector             ( descriptorHist[0][0]           )
{ }

/**** Setters ************************************************************************/

void Detector::setUpscale(bool upscale) {
	this->upscale = upscale;
}

void Detector::setBorderWidth(int width) {
	if (width < 0)
		throw domain_error("The border width must be positive or zero.");

	this->borderWidth = width;
}

void Detector::setIntervals(int intervals) {
	if (intervals < INTERVALS_MIN)
		throw domain_error("too few intervals");

	this->intervals = intervals;
}

void Detector::setSigma(float sigma) {
	if (sigma <= 0.f)
		throw domain_error("sigma must be positive");

	this->sigma = sigma;
}

void Detector::setSigmaInit(float sigmaInit) {
	if (sigmaInit < 0.f)
		throw domain_error("sigmaInit must be positive or zero");

	this->sigmaInit = sigmaInit;
}

void Detector::setContrastThreshold(float threshold) {
	if (threshold <= 0.f)
		throw domain_error("The contrast threshold must be positive.");

	this->contrastThreshold = threshold;
}

void Detector::setCurvatureThreshold(float threshold) {
	if (threshold <= 0.f)
		throw domain_error("The curvature threshold must be positive.");

	this->curvatureThreshold = threshold;
}

void Detector::setOrientationSigmaFactor(float factor) {
	if (factor <= 0.f)
		throw domain_error("The orientation sigma factor must be positive.");

	this->orientationSigmaFactor = factor;
}

void Detector::setOrientationRadiusFactor(float factor) {
	if (factor <= 0.f)
		throw domain_error("The orientation radius factor must be positive.");

	this->orientationRadiusFactor = factor;
}

void Detector::setOrientationPeakRatio(float ratio) {
	if (ratio < 0.f || ratio > 1.f)
		throw domain_error("The orientation peak ratio must be in [0, 1].");

	this->orientationPeakRatio = ratio;
}

void Detector::setOrientationSmoothIterations(int iterations) {
	if (iterations < 0)
		throw domain_error(
			"The amount of orientation smoothing iterations must be positive or zero.");

	this->orientationSmoothIterations = iterations;
}

void Detector::setDescriptorSigmaFactor(float factor) {
	if (factor <= 0.f)
		throw domain_error("The descriptor sigma factor must be positive.");

	this->descriptorSigmaFactor = factor;
}

void Detector::setDescriptorMagnitudeThreshold(float threshold) {
	if (threshold <= 0.f || threshold >= 1.f)
		throw domain_error("The descriptor magnitude threshold must be in (0, 1).");

	this->descriptorMagnitudeThreshold = threshold;
}

/**** Public Methods *****************************************************************/

int Detector::detect(const Mat &img, std::vector<KeyPoint> &keyPoints, cv::Mat &descriptors) const {
	init(img);
	loop();
	buildFeatures(keyPoints, descriptors);
	cleanUp();

	return keyPoints.size();
}

int Detector::detect(const Mat &img, Model &model) const {
	return model.detect(*this, img);
}

/**** Private Methods ****************************************************************/

void Detector::init(const Mat &img) const {
	// convert image to grayscale floating point matrix
	Mat gray;
	if (img.channels() == 3)
		cvtColor(img, gray, CV_RGB2GRAY);
	else if (img.channels() == 1)
		gray = img;
	else
		throw BadImageFormat("The image must either have three channels or one.", img);

	gray.convertTo(gray, CV_32F, 1. / UCHAR_MAX);

	Size size = gray.size();

	width   = size.width;
	height  = size.height;
	// S := log2(m) - 2
	// w_s := 2^-s * w_0 => w_s := 4 / m * w_0
	// where S := octaves, w_s is the width or height of the s-th octave and m := min(width, height)
	octaves = cvFloor(log((float) min(width, height)) / log(2.f) - 2.f);

	pyramid.start(gray, upscale, intervals, sigma, sigmaInit);
}

void Detector::cleanUp() const {
	candidates.clear();
	keyPoints.clear();
	descriptors.clear();
	pyramid.clear();
}

void Detector::loop() const {
	while (pyramid.getOctave() < octaves)
		step();
}

void Detector::step() const {
	Size size = pyramid.getCurrentSize();

	width        = size.width;
	height       = size.height;
	rightBorder  = size.width  - borderWidth;
	bottomBorder = size.height - borderWidth;
	interval     = pyramid.getInterval();
	octave       = pyramid.getOctave();

	detectCandidates();
	calcFeatures();

	// last step unnecessary
	pyramid.step();
}

void Detector::buildFeatures(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptors) const {
	int n = this->keyPoints.size();

	keyPoints = vector<KeyPoint>(this->keyPoints.begin(), this->keyPoints.end());
	descriptors = Mat::zeros(n, DESCRIPTOR_LENGTH, CV_32F);

	// write each descriptor into descriptors
	int i = 0;
	for each (const Mat d in this->descriptors)
		descriptors.row(i++) += d;
}

void Detector::detectCandidates() const {
	float prelimContrastThreshold = contrastThreshold / (2*intervals);
	const Mat &curr = pyramid.getCurrentDog();

	for (int c = borderWidth; c < rightBorder; ++c) {
		for (int r = borderWidth; r < bottomBorder; ++r) {
			float value = curr.at<float>(r, c);

			if (abs(value) <= prelimContrastThreshold)
				continue;
			if (!isExtremum(c, r))
				continue;
			
			Candidate candidate;
			candidate.c = c;
			candidate.r = r;

			if (!interpolateExtremum(candidate))
				continue;
			if (isEdge(candidate.c, candidate.r))
				continue;

			float xi = candidate.offset.at<float>(CANDIDATE_OFFSET_XI);

			candidate.octaveScale = sigma * pow(2.f, (interval + xi) / intervals);
			candidate.histSize    = descriptorSigmaFactor * candidate.octaveScale;
			// radius := ( \sqrt{2} * (n + 1) * h + 1 ) / 2
			// where n := DESCRIPTOR_HIST_WIDTH and h = histSize
			candidate.radius      = sqrt(.5f) * (DESCRIPTOR_HIST_WIDTH + 1) * candidate.histSize + .5f;

			calcOrientedCandidates(candidate);
		}
	}
}

bool Detector::isExtremum(int c, int r) const {
	const Mat &prev = pyramid.getPreviousDog();
	const Mat &curr = pyramid.getCurrentDog();
	const Mat &next = pyramid.getNextDog();

	float val   = curr.at<float>(r    , c    );
	float first = prev.at<float>(r - 1, c - 1);

	if (first > val) {
		// check minimum
		// for each adjacent pixel
		for (int dc = -1; dc <= 1; ++dc) {
			for (int dr = -1; dr <= 1; ++dr) {
				float padj = prev.at<float>(r + dr, c + dc);
				float cadj = curr.at<float>(r + dr, c + dc);
				float nadj = next.at<float>(r + dr, c + dc);

				if (val >= padj || val >= nadj || val > cadj)
					return false;
				if (val == cadj && dc != 0 && dr != 0)
					return false;
			}
		}

		return true;
	} else if (first < val) {
		// check maximum
		// for each adjacent pixel
		for (int dc = -1; dc <= 1; ++dc) {
			for (int dr = -1; dr <= 1; ++dr) {
				float padj = prev.at<float>(r + dr, c + dc);
				float cadj = curr.at<float>(r + dr, c + dc);
				float nadj = next.at<float>(r + dr, c + dc);

				if (val <= padj || val <= nadj || val < cadj)
					return false;
				if (val == cadj && dc != 0 && dr != 0)
					return false;
			}
		}

		return true;
	}

	return false;
}

bool Detector::isEdge(int c, int r) const {
	const Mat &curr = pyramid.getCurrentDog();

	float val = curr.at<float>(r, c);

	// compute 2x2 Hessian
	float dxx = curr.at<float>(r    , c + 1) + curr.at<float>(r    , c - 1) - 2.f * val;
	float dyy = curr.at<float>(r + 1, c    ) + curr.at<float>(r - 1, c    ) - 2.f * val;
	float dxy = (
		curr.at<float>(r + 1, c + 1) - curr.at<float>(r + 1, c - 1) -
		curr.at<float>(r - 1, c + 1) + curr.at<float>(r - 1, c - 1) ) / 4.f;

	float trace = dxx + dyy;
	float det   = dxx * dyy - dxy * dxy;

	// negative determinant -> curvatures have different signs; reject feature
	if (det <= 0)
		return true;

	// trace^2 / det < (c + 1)^2 / c
	// where c := curvatureThreshold
	return trace*trace / det >= curvatureThreshold + 2.f + 1.f/curvatureThreshold;
}

bool Detector::interpolateExtremum(Candidate &candidate) const {
	// note that c and r can be changed by interpolateOffset
	int &c = candidate.c;
	int &r = candidate.r;

	// interpolate (c, r) coordinates
	int i;
	for (i = 0; i < MAX_INTERPOLATION_STEPS; ++i) {
		// if offset too little jump out
		if (!interpolateOffset(candidate)) // changes c and r
			break;
		
		float xi    = candidate.offset.at<float>(CANDIDATE_OFFSET_XI);
		int   intvl = interval + cvRound(xi);

		// if not within image borders or octave
		if (c     < borderWidth || c     >= rightBorder  ||
			r     < borderWidth || r     >= bottomBorder ||
			intvl < 0           || intvl >= intervals )
		{
			return false;
		}
	}

	// if interpolation didn't converged reject
	if (i >= MAX_INTERPOLATION_STEPS)
		return false;

	float contrast = interpolateContrast(candidate);

	// if low contrast reject
	if (abs(contrast) < contrastThreshold / intervals)
		return false;

	return true;
}

bool Detector::interpolateOffset(Candidate &candidate) const {
	Mat dD   = derive3D(candidate.c, candidate.r);
	Mat hInv = calcHessian3D(candidate.c, candidate.r).inv(CV_SVD);

	candidate.offset = - hInv.t() * dD;

	float xc = candidate.offset.at<float>(CANDIDATE_OFFSET_XC);
	float xr = candidate.offset.at<float>(CANDIDATE_OFFSET_XR);
	float xi = candidate.offset.at<float>(CANDIDATE_OFFSET_XI);

	if (abs(xc) < .5f && abs(xr) < .5f && abs(xi) < .5f)
		return false;

	candidate.c += cvRound(xc);
	candidate.r += cvRound(xr);

	return true;
}

float Detector::interpolateContrast(const Candidate &candidate) const {
	int c = candidate.c;
	int r = candidate.r;
	const Mat &curr = pyramid.getCurrentDog();

	float val = curr.at<float>(r, c);
	Mat dD       = derive3D(c, r);
	Mat contrast = dD.t() * candidate.offset;
	
	// val + |D(\deltax)| / 2
	return val + contrast.at<float>() / 2.f;
}

void Detector::calcOrientedCandidates(const Candidate &candidate) const {
	calcOrientationHist(candidate);

	// allow only magnitudes similar to the dominant orientation
	float magnitudeThreshold = orientationPeakRatio * findDominantOrientation();

	// for each bin
	for (int i = 0; i < ORIENTATION_BINS; ++i) {
		// left and right indices
		int l = (i == 0) ? ORIENTATION_BINS - 1 : i - 1;
		int r = (i == ORIENTATION_BINS - 1) ? 0 : i + 1;

		// left, curr and right bins
		float left  = orientationHist[l];
		float curr  = orientationHist[i];
		float right = orientationHist[r];

		// unless bin is local extrema and satisfies magnitude threshold
		if (curr <= left || curr <= right || curr < magnitudeThreshold)
			continue;

		// interpolation range is (-0.5, 0.5)
		float interpolation = 0.5f * (left - right) / (left - 2.f*curr + right);
		// orientation range is (-PI / n, 2PI + PI / n) where n := ORIENTATION_BINS
		float orientation   = (2.f*PI_F * (i + interpolation)) / ORIENTATION_BINS;

		// normalize range to (-PI, PI]
		if (orientation > PI_F)
			orientation -= 2.f*PI_F;

		// copy into candidate list
		candidates.push_back(candidate);
		// get new copy
		Candidate &newCandidate = candidates.back();
		// save orientation
		newCandidate.orientation = orientation;
	}
}

void Detector::calcOrientationHist(const Candidate &candidate) const {
	resetOrientationHist();

	int   radius           = cvRound(orientationRadiusFactor * candidate.octaveScale);
	float orientationSigma = orientationSigmaFactor * candidate.octaveScale;

	// for each pixel within radius
	float magnitude, angle;
	for (int dc = -radius; dc <= radius; ++dc) {
		for (int dr = -radius; dr <= radius; ++dr) {
			// if gradient exists (within the image)
			if (!calcGradient(candidate.c + dc, candidate.r + dr, magnitude, angle))
				continue;

			int bin = cvRound(ORIENTATION_BINS * angle / (2.f*PI_F));

			// assure positive bin (from 0 to 2 PI)
			if (bin < 0)
				bin += ORIENTATION_BINS;

			float weight = exp( -(dc*dc + dr*dr) / (2.f * orientationSigma*orientationSigma) );

			orientationHist[bin] += weight * magnitude;
		}
	}

	// smooth histogram multiple times
	for (int i = 0; i < orientationSmoothIterations; ++i)
		smoothOrientationHist();
}

void Detector::smoothOrientationHist() const {
	// temporarly save original values from being lost
	float first = orientationHist[0]; // first bin
	float prev  = orientationHist[ORIENTATION_BINS - 1]; // last bin

	// for each bin
	for (int bin = 0; bin < ORIENTATION_BINS; ++bin) {
		float curr = orientationHist[bin];
		// regard wrap around
		float next = (bin + 1 == ORIENTATION_BINS) ? first : orientationHist[bin + 1];

		orientationHist[bin] = .25f*prev + .5f*curr + .25f*next;

		prev = curr;
	}
}

float Detector::findDominantOrientation() const {
	// current maximum
	float maximum = orientationHist[0]; // first bin
	
	// for each bin
	for (int bin = 0; bin < ORIENTATION_BINS; ++bin) {
		// if greater than current maximum
		if (orientationHist[bin] > maximum)
			maximum = orientationHist[bin];
	}

	return maximum;
}

void Detector::resetOrientationHist() const {
	// reset entries to 0.f
	memset(orientationHist, 0, sizeof(orientationHist));
}

void Detector::calcFeatures() const {
	// for each candidate
	// calculate key point and descriptor
	while (!candidates.empty()) {
		const Candidate &c = candidates.front();

		KeyPoint k;
		// descriptor
		Mat d(1, DESCRIPTOR_LENGTH, CV_32F);

		calcKeyPoint(c, k);
		calcDescriptorMat(c, d);

		keyPoints.push_back(k);
		descriptors.push_back(d);

		candidates.pop_front();
	}
}

void Detector::calcKeyPoint(const Candidate &candidate, cv::KeyPoint &keyPoint) const {
	float xc = candidate.offset.at<float>(CANDIDATE_OFFSET_XC);
	float xr = candidate.offset.at<float>(CANDIDATE_OFFSET_XR);
	float xi = candidate.offset.at<float>(CANDIDATE_OFFSET_XI);

	float x = (candidate.c + xc) * pow(2.f, octave);
	float y = (candidate.r + xr) * pow(2.f, octave);

	keyPoint.pt     = Point2f(x, y);
	keyPoint.angle  = candidate.orientation * 180.f / PI_F;
	keyPoint.octave = octave;
	keyPoint.size   = sigma * pow(2.f, octave + (interval + xi) / intervals);
}

void Detector::calcDescriptorMat(const Candidate &candidate, Mat &descriptor) const {
	calcDescriptor(candidate);
	convertDescriptor2Mat(descriptor);
}

void Detector::calcDescriptor(const Candidate &candidate) const {
	resetDescriptor();

	float orientation = candidate.orientation;
	int   radius      = (int) candidate.radius;
	float cosT        = cos(orientation);
	float sinT        = sin(orientation);

	// for each pixel within radius
	float magnitude, angle;
	for (int dc = -radius; dc <= radius; ++dc) {
		for (int dr = -radius; dr <= radius; ++dr) {
			// rotate coordinates for orientation normalization
			// normalized to sub histogram width
			float dcRot = (dc * cosT - dr * sinT) / candidate.histSize;
			float drRot = (dc * sinT + dr * cosT) / candidate.histSize;
			
			// calculate bin indices
			// bins are centered within sub histograms
			// \deltax_Hist := dx_rot + (n - 1) / 2
			// where n := DESCRIPTOR_HIST_WIDTH and x \in {d, c}
			float dcHist = dcRot + (DESCRIPTOR_HIST_WIDTH / 2) - .5f;
			float drHist = drRot + (DESCRIPTOR_HIST_WIDTH / 2) - .5f;

			// if outside of histogram range
			if (dcHist <= -1.f || dcHist >= DESCRIPTOR_HIST_WIDTH || drHist <= -1.f || drHist >= DESCRIPTOR_HIST_WIDTH)
				continue;

			// if gradient exists (within the image)
			if (calcGradient(candidate.c + dc, candidate.r + dr, magnitude, angle)) {
				// normalize angle to orientation
				// angle and orientation range is (-PI, PI]
				// therefore oRot range is (-2PI, 2PI)
				float oRot = angle - orientation;

				// normalize to positive value [0, 2 PI)
				if (oRot < 0.f)
					oRot += 2.f * PI_F;
				// considering top range evaluation '>' comparision could be enough
				// playing safe because of round-off error
				else if (oRot >= 2.f * PI_F)
					oRot -= 2.f * PI_F;

				float oBin   = oRot * DESCRIPTOR_HIST_BINS / (2.f * PI_F);
				float weight = exp( -(dcRot*dcRot + drRot*drRot) / (.5f * DESCRIPTOR_HIST_WIDTH*DESCRIPTOR_HIST_WIDTH) );
				
				distributeDescriptorEntry(dcHist, drHist, oBin, weight * magnitude);
			}
		}
	}

	normalizeAndTrimDescriptor();
}

void Detector::convertDescriptor2Mat(Mat &descriptor) const {
	// for each descriptor entry
	for (int i = 0; i < DESCRIPTOR_LENGTH; ++i) {
		float val = DESCRIPTOR_FLOAT2UCHAR_FACTOR * descriptorVector[i];
		descriptor.at<float>(i) = min(val, (float) UCHAR_MAX);
	}
}

void Detector::distributeDescriptorEntry(float dcHist, float drHist, float oBin, float magnitude) const {
	// sub histogram left from sample point
	int dcHist0 = cvFloor(dcHist);
	// sub histogram above sample point
	int drHist0 = cvFloor(drHist);
	// histogram bin 'left' from sampled orientation
	int oBin0   = cvFloor(oBin);

	// distance of sample to left sub histogram's center
	float ddcHist0 = dcHist - dcHist0;
	// distance of sample to top sub histogram's center
	float ddrHist0 = drHist - drHist0;
	// distance of sampled orientation to 'left' histogram bin's left border
	float doBin0   = oBin - oBin0;
	// distance of sample to right sub histogram's center
	float ddcHist1 = 1.f - ddcHist0;
	// distance of sample to bottom sub histogram's center
	float ddrHist1 = 1.f - ddrHist0;
	// distance of sampled orientation to 'right' histogram bin's left border
	float doBin1   = 1.f - doBin0;

	// destribute gradient to subhistogram neihbors and orientation bin neighbors
	// weight initial magnitude by distance (1 - d) to original bin
	
	// for vertical adjacent sub histograms
	for (int ddrAdj = 0; ddrAdj <= 1; ++ddrAdj) {
		// current adjacent vertical sub histogram
		int drAdj = drHist0 + ddrAdj;

		// if outside of region
		if (drAdj < 0 || drAdj >= (int) DESCRIPTOR_HIST_WIDTH)
			continue;

		// vertical weighted magnitude
		float vR = (ddrAdj == 0 ? ddrHist1 : ddrHist0) * magnitude;

		// for horizontal adjacent sub histograms
		for (int ddcAdj = 0; ddcAdj <= 1; ++ddcAdj) {
			// current adjacent horizontal sub histogram
			int dcAdj = dcHist0 + ddcAdj;

			// if outside of region
			if (dcAdj < 0 || dcAdj >= (int) DESCRIPTOR_HIST_WIDTH)
				continue;

			// horizontal weighted magnitude
			float vC = (ddcAdj == 0 ? ddcHist1 : ddcHist0) * vR;

			// for adjacent orientation bin
			for (int doAdj = 0; doAdj <= 1; ++doAdj) {
				// current adjacent orientation bin (with wrap around)
				int oAdj = (oBin0 + doAdj) % DESCRIPTOR_HIST_BINS;
				// orientation weighted magnitude
				float vO = (doAdj == 0 ? doBin1 : doBin0) * vC;

				descriptorHist[drAdj][dcAdj][oAdj] += vO;
			}
		}
	}
}

void Detector::normalizeDescriptor() const {
	float squareLength = 0.;

	// sum square length over all descriptor entries
	for (int i = 0; i < DESCRIPTOR_LENGTH; ++i) {
		float curr = descriptorVector[i];
		squareLength += curr*curr;
	}

	float inverseLength = 1.f / sqrt(squareLength);

	// normalize each descriptor entry
	for (int i = 0; i < DESCRIPTOR_LENGTH; ++i)
		descriptorVector[i] *= inverseLength;
}

bool Detector::trimDescriptor() const {
	// for each descriptor vector's element
	bool trimmed = false;
	for (int i = 0; i < DESCRIPTOR_LENGTH; ++i) {
		// if above magnitude threshold
		if (descriptorVector[i] > descriptorMagnitudeThreshold) {
			// trim to threshold
			descriptorVector[i] = descriptorMagnitudeThreshold;
			trimmed = true;
		}
	}

	return trimmed;
}

void Detector::normalizeAndTrimDescriptor() const {
	normalizeDescriptor();

	// if descriptor needed to be trimmed
	if (trimDescriptor())
		normalizeDescriptor();
}

void Detector::resetDescriptor() const {
	// reset entries to 0.f
	memset(descriptorHist, 0, sizeof(descriptorHist));
}

bool Detector::calcGradient(int c, int r, float &magnitude, float &angle) const {
	// if gradient is not available
	if (c <= 0 || c >= width - 1 || r <= 0 || r >= height - 1)
		return false;

	const Mat &curr = pyramid.getCurrentSs();

	float dx = curr.at<float>(r    , c + 1) - curr.at<float>(r    , c - 1);
	float dy = curr.at<float>(r - 1, c    ) - curr.at<float>(r + 1, c    );

	magnitude = sqrt(dx*dx + dy*dy);
	angle     = atan2(dy, dx);

	return true;
}

Mat Detector::derive3D(int c, int r) const {
	const Mat &prev = pyramid.getPreviousDog();
	const Mat &curr = pyramid.getCurrentDog();
	const Mat &next = pyramid.getNextDog();

	float dc = (curr.at<float>(r    , c + 1) - curr.at<float>(r    , c - 1)) / 2.f;
	float dr = (curr.at<float>(r + 1, c)     - curr.at<float>(r - 1, c    )) / 2.f;
	float ds = (next.at<float>(r    , c)     - prev.at<float>(r    , c    )) / 2.f;

	// return [dc dr ds]^T
	return Mat_<float>(3, 1) << dc, dr, ds;
}

Mat Detector::calcHessian3D(int c, int r) const {
	const Mat &prev = pyramid.getPreviousDog();
	const Mat &curr = pyramid.getCurrentDog();
	const Mat &next = pyramid.getNextDog();

	float val = curr.at<float>(r, c);

	float dxx = curr.at<float>(r    , c + 1) + curr.at<float>(r    , c - 1) - 2.f * val;
	float dyy = curr.at<float>(r + 1, c    ) + curr.at<float>(r - 1, c    ) - 2.f * val;
	float dss = next.at<float>(r    , c    ) + prev.at<float>(r    , c    ) - 2.f * val;
	float dxy = (
		curr.at<float>(r + 1, c + 1) - curr.at<float>(r + 1, c - 1) -
		curr.at<float>(r - 1, c + 1) + curr.at<float>(r - 1, c - 1) ) / 4.f;
	float dxs = (
		next.at<float>(r    , c + 1) - next.at<float>(r    , c - 1) -
		prev.at<float>(r    , c + 1) + prev.at<float>(r    , c - 1) ) / 4.f;
	float dys = (
		next.at<float>(r + 1, c    ) - next.at<float>(r - 1, c    ) -
		prev.at<float>(r + 1, c    ) + prev.at<float>(r - 1, c    ) ) / 4.f;

	return Mat_<float>(3, 3) << dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss;
}
