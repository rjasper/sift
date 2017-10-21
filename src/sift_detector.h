#ifndef SIFT_DETECTOR_H
#define SIFT_DETECTOR_H

#include <list>

#include <opencv2/opencv.hpp>

#include "sift.h"
#include "sift_candidate.h"
#include "sift_ss_dog_pyramid.h"


namespace sift {

class Model;

///=================================================================================================
/// <summary>
/// The Detector extracts SIFT-Features from images. Several SIFT-Parameters can be configured
/// before Detection.
/// </summary>
/// 
/// <remarks> Jasper, 15.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class Detector {

public:

/******************************************************************************************************* 
 ****** 0. Constructors ********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Default constructor. Sets default sift parameters.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	Detector();
		
/******************************************************************************************************* 
 ****** I. Public Methods ******************************************************************************
 ****** I.a Setters       ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Sets whether an initial upscale will be performed. If set to true an additional octave "-1" is
	/// calculated. Width and height are doubled. If set to false the calculations begin in the 0th
	/// octave with original width and height.
	/// </summary>
	/// 
	/// <param name="upscale"> true to scale up. </param>
	///-------------------------------------------------------------------------------------------------
	void setUpscale(bool upscale);

	///=================================================================================================
	/// <summary>
	/// Sets the border width in which to ignore detected extrema.
	/// </summary>
	/// 
	/// <param name="width"> The width. (width >= 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setBorderWidth(int width);

	///=================================================================================================
	/// <summary>
	/// Sets the number of sampled intervals per octave.
	/// </summary>
	/// 
	/// <param name="intervals">
	/// The number intervals an octave shall consist of. (intervals > INTERVALS_MIN)
	/// </param>
	///-------------------------------------------------------------------------------------------------
	void setIntervals(int intervals);

	///=================================================================================================
	/// <summary>
	/// Sets the standard deviation of the gaussian bluring which is doubled each octave.
	/// </summary>
	/// 
	/// <param name="sigma"> The standard deviation of each interval blurring. (sigma > 0)</param>
	///-------------------------------------------------------------------------------------------------
	void setSigma(float sigma);

	///=================================================================================================
	/// <summary>
	/// Sets the assumed standard deviation of the gaussian blur for the input image.
	/// </summary>
	/// 
	/// <param name="sigmaInit">
	/// The standard deviation of the initial input image blurring. (sigmaInit >= 0)
	/// </param>
	///-------------------------------------------------------------------------------------------------
	void setSigmaInit(float sigmaInit);
	
	///=================================================================================================
	/// <summary>
	/// Sets the contrast threshold. Key points with lower contrast are discarded.
	/// </summary>
	/// 
	/// <param name="threshold"> The threshold. (threshold > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setContrastThreshold(float threshold);

	///=================================================================================================
	/// <summary>
	/// Sets the curvature threshold of principle curvatures used for edge detection. In general the
	/// smaller the threshold the more sensitive the edge detection.
	/// </summary>
	/// 
	/// <param name="threshold"> The threshold. (threshold > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setCurvatureThreshold(float threshold);

	///=================================================================================================
	/// <summary>
	/// Sets the orientation sigma factor which is used to calculate the
	/// gaussian weight for the orientation histogram. The octave scale is
	/// multiplied by the factor resulting in the orientation sigma.
	/// </summary>
	/// 
	/// <param name="factor"> The factor. (factor > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationSigmaFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the orientation radius factor which is used to calculate the region size for the
	/// orientation assignment. The radius of the region is determined by multiplying the octave scale
	/// by the factor.
	/// </summary>
	/// 
	/// <param name="factor"> The factor. (factor > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationRadiusFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the orientation peak ratio. An orientation results in a new feature if its
	/// magnitude-maximum ratio is at least as high as the given one.
	/// </summary>
	/// 
	/// <param name="ratio"> The ratio. (ratio \in [0, 1]) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationPeakRatio(float ratio);

	///=================================================================================================
	/// <summary>
	/// Sets the number of smooth iterations of the orientation histogramm.
	/// </summary>
	/// 
	/// <param name="iterations"> The number of iterations. (iterations >= 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setOrientationSmoothIterations(int iterations);

	///=================================================================================================
	/// <summary>
	/// Sets the descriptor sigma factor which is used to calculate the size of a descriptor sub
	/// histogram relative to the octave scale.
	/// </summary>
	/// 
	/// <param name="factor"> The factor. (factor > 0) </param>
	///-------------------------------------------------------------------------------------------------
	void setDescriptorSigmaFactor(float factor);

	///=================================================================================================
	/// <summary>
	/// Sets the magnitude threshold for the descriptor vector after normalization.
	/// </summary>
	/// 
	/// <param name="threshold"> The threshold. (factor \in (0, 1)) </param>
	///-------------------------------------------------------------------------------------------------
	void setDescriptorMagnitudeThreshold(float threshold);
		
/******************************************************************************************************* 
 ****** I.b Use Case Methods ***************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Detects SIFT features in the given input image. The key points and the descriptors are stored in
	/// the two given vectors. A key point descriptor pair will have the same index in their vectors.
	/// </summary>
	/// 
	/// <param name="img"> The input image. </param>
	/// <param name="keyPoints"> [in, out] The vector in which to store the key points. </param>
	/// <param name="descriptors"> [in, out] The vector in which to store the descriptors. </param>
	///
	/// <returns> The number of found features. </returns>
	///-------------------------------------------------------------------------------------------------
	int detect(const cv::Mat &img, std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptors) const;

	///=================================================================================================
	/// <summary>
	/// Detects SIFT features in the given input image. The key points and the descriptors are stored in
	/// the given model.
	/// </summary>
	/// 
	/// <param name="img"> The input image. </param>
	/// <param name="model"> [in, out] The model in which to store the key points and descriptors. </param>
	///
	/// <returns> The number of found features. </returns>
	///-------------------------------------------------------------------------------------------------
	int detect(const cv::Mat &img, sift::Model &model) const;

private:
		
/******************************************************************************************************* 
 ****** II. Fields    **********************************************************************************
 ****** II.a Settings **********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Stores whether an initial upscale will be performed. If set to true an additional octave "-1" is
	/// calculated. Width and height are doubled. If set to false the calculations begin in the 0th
	/// octave with original width and height.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	bool upscale;

	///=================================================================================================
	/// <summary>
	/// The border width in which to ignore detected extrema.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int borderWidth;

	///=================================================================================================
	/// <summary>
	/// The number of sampled intervals per octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int intervals;

	///=================================================================================================
	/// <summary>
	/// The standard deviation of the gaussian bluring which is doubled each octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float sigma;

	///=================================================================================================
	/// <summary>
	/// The assumed standard deviation of the gaussian blur for the input image.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float sigmaInit;

	///=================================================================================================
	/// <summary>
	/// The contrast threshold. Key points with lower contrast are discarded.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float contrastThreshold;

	///=================================================================================================
	/// <summary>
	/// The curvature threshold of principle curvatures used for edge detection. In general the smaller
	/// the threshold the more sensitive the edge detection.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float curvatureThreshold;

	///=================================================================================================
	/// <summary>
	/// The orientation sigma factor which is used to calculate the gaussian weight for the orientation
	/// histogram. The octave scale is multiplied by the factor resulting in the orientation sigma.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float orientationSigmaFactor;

	///=================================================================================================
	/// <summary>
	/// The orientation radius factor which is used to calculate the region size for the orientation
	/// assignment. The radius of the region is determined by multiplying the octave scale by the
	/// factor.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float orientationRadiusFactor;

	///=================================================================================================
	/// <summary>
	/// The orientation peak ratio. An orientation results in a new feature if its magnitude-maximum
	/// ratio is at least as high as the given one.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float orientationPeakRatio;

	///=================================================================================================
	/// <summary>
	/// Sets the number of smooth iterations of the orientation histogramm.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int orientationSmoothIterations;

	///=================================================================================================
	/// <summary>
	/// The descriptor sigma factor which is used to calculate the size of a descriptor sub histogram
	/// relative to the octave scale.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float descriptorSigmaFactor;

	///=================================================================================================
	/// <summary>
	/// The magnitude threshold for the descriptor vector after normalization.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float descriptorMagnitudeThreshold;
	
/******************************************************************************************************* 
 ****** II.b State               ***********************************************************************
 ****** II.b (1) Detection State ***********************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The amount of octaves to be calculated. If upscale is set the amount of octaves is octaves + 1.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int octaves;

	///=================================================================================================
	/// <summary>
	/// The list of key points found during detection.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable std::list<cv::KeyPoint> keyPoints;

	///=================================================================================================
	/// <summary>
	/// The list of descriptors found during detection.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable std::list<cv::Mat> descriptors;

	///=================================================================================================
	/// <summary>
	/// The scale space and difference of gaussian pyramid of the input image. Features are extracted
	/// from the pyramid.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable SsDogPyramid pyramid;
	
/******************************************************************************************************* 
 ****** II.b (2) Octave State ************************************************************************** 
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The current octave features are extracted from.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int octave;

	///=================================================================================================
	/// <summary>
	/// The height of the current octave's interval images.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int height;

	///=================================================================================================
	/// <summary>
	/// The width of the current octave's interval images.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int width;

	///=================================================================================================
	/// <summary>
	/// The right border's position of the current octave's interval images.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int rightBorder;

	///=================================================================================================
	/// <summary>
	/// The bottom border's position of the current octave's interval images.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int bottomBorder;

/******************************************************************************************************* 
 ****** II.b (3) Interval State ************************************************************************ 
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The current interval features are extracted from.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable int interval;

	///=================================================================================================
	/// <summary>
	/// The list of candidates found during the current interval.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable std::list<Candidate> candidates;

/******************************************************************************************************* 
 ****** II.b (4) Feature State *************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The orientation histogram of the feature currently being extracted.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable float orientationHist[ORIENTATION_BINS];

	///=================================================================================================
	/// <summary>
	/// The descriptor, represented as multiple orientation histograms, of the the feature currently
	/// being extracted.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable float descriptorHist[DESCRIPTOR_HIST_WIDTH][DESCRIPTOR_HIST_WIDTH][DESCRIPTOR_HIST_BINS];
	
	///=================================================================================================
	/// <summary>
	/// The descriptor, represented as a vector, of the feature currently being extracted. Serves as a
	/// helper since it points to descriptorHist, which holds the actual descriptor data.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	mutable float *descriptorVector;
	
/******************************************************************************************************* 
 ****** III. Private Methodes          *****************************************************************
 ****** III.a Initializer and Clean Up *****************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Initializes the detection process. Allocates memory.
	/// </summary>
	/// 
	/// <param name="img"> The image. </param>
	///-------------------------------------------------------------------------------------------------
	void init(const cv::Mat &img) const;

	///=================================================================================================
	/// <summary>
	/// Deallocates memory needed for the detection process.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void cleanUp() const;

/******************************************************************************************************* 
 ****** III.b SIFT Algorithm    ************************************************************************
 ****** III.b (1) Interval step ************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Performs multiple steps until the final octave has been reached.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void loop() const;

	///=================================================================================================
	/// <summary>
	/// Performs a detection step. Detects candidates and calculates features of the current interval.
	/// Afterwards the pyramid calculates the next interval.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void step() const;
	
	///=================================================================================================
	/// <summary>
	/// Builds a key point vector and descriptor matrix from the extracted features.
	/// </summary>
	/// 
	/// <param name="keyPoints"> [in,out] The key points. </param>
	/// <param name="descriptors"> [in,out] The descriptors. </param>
	///-------------------------------------------------------------------------------------------------
	void buildFeatures(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &descriptors) const;

/*******************************************************************************************************
 ****** III.b (2) Candidate Detection ******************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Detects candidates in the current interval image. Found candidates are stored in the candidates
	/// list and are ready for feature calculations.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void detectCandidates() const;

	///=================================================================================================
	/// <summary>
	/// Queries if the point (c, r) is an extremum. The point is considered as extremum if either all 26
	/// adjacent pixel (including previous and next interval) are lower or greater than the point's
	/// value.
	/// </summary>
	/// 
	/// <param name="c"> The pixel column. </param>
	/// <param name="r"> The pixel row. </param>
	/// 
	/// <returns> true if extremum, false if not. </returns>
	///-------------------------------------------------------------------------------------------------
	bool isExtremum(int c, int r) const;

	///=================================================================================================
	/// <summary>
	/// Queries if the point (c, r) is an edge. The point is considered as an edge if the principle
	/// curvatures ratio is above the curvature threshold.
	/// </summary>
	/// 
	/// <param name="c"> The pixel column. </param>
	/// <param name="r"> The pixel row. </param>
	/// 
	/// <returns> true if edge, false if not. </returns>
	///-------------------------------------------------------------------------------------------------
	bool isEdge(int c, int r) const;

	///=================================================================================================
	/// <summary>
	/// <para>Interpolates the extremum position of the given candidate. Therefore the position of the
	/// candidate might be changed. The subpixel offset is stored within the candidate.</para>
	/// <para>The interpolation process (3D-quadratic fit) operates on the current interval plane. The
	/// offset of the candidate iterates multiple times until either repositioning doesn't do any effect
	/// (values < .5) or the maximum number of iterations has been reached. In the last case the
	/// candidate will be rejected since the interpolation didn't converged. If it passed the contrast
	/// is interpolated at the new position including the offset. The candidate will be accepted if the
	/// contrast is above the contrast threshold.</para>
	/// </summary>
	/// 
	/// <param name="candidate"> [in,out] The candidate. </param>
	/// 
	/// <returns> true if the candidate was accepted, false if rejected. </returns>
	///-------------------------------------------------------------------------------------------------
	bool interpolateExtremum(Candidate &candidate) const;

	///=================================================================================================
	/// <summary>
	/// Interpolates the subpixel offset of the given candidate's position. The offset is calculated by
	/// fitting a 3D-quadratic function from which the vertex' position is taken as subpixel extremum
	/// position.
	/// </summary>
	/// 
	/// <param name="candidate"> [in,out] The candidate. </param>
	/// 
	/// <returns> true if the candidate's position was adjusted to the
	/// subpixel offset. </returns>
	///-------------------------------------------------------------------------------------------------
	bool interpolateOffset(Candidate &candidate) const;

	///=================================================================================================
	/// <summary>
	/// Interpolates the contrast at the offset position of the given candidate.
	/// </summary>
	/// 
	/// <param name="candidate"> [in,out] The candidate. </param>
	/// 
	/// <returns> the interpolated contrast. </returns>
	///-------------------------------------------------------------------------------------------------
	float interpolateContrast(const Candidate &candidate) const;

/*******************************************************************************************************
 ****** III.b (3) Orientation Assignment ***************************************************************
 *******************************************************************************************************/
	
	///=================================================================================================
	/// <summary>
	/// Calculates oriented versions of the given candidate to the candidate list. For each orientation
	/// satisfying the orientationPeakRatio a copy of the given candidate is created. The orientation is
	/// assigned to the copy. Eventually the copy is stored in the candidates list.
	/// </summary>
	/// 
	/// <param name="candidate"> The candidate. </param>
	///-------------------------------------------------------------------------------------------------
	void calcOrientedCandidates(const Candidate &candidate) const;

	///=================================================================================================
	/// <summary>
	/// Calculates the orientation histogram of the given candidate. The histogram contains the gaussian
	/// weighted magnitudes of gradients in dependancy to their angles. Afterwards the histogram is
	/// smoothed multiple times.
	/// </summary>
	/// 
	/// <param name="candidate"> The candidate. </param>
	///-------------------------------------------------------------------------------------------------
	void calcOrientationHist(const Candidate &candidate) const;

	///=================================================================================================
	/// <summary>
	/// Smooths orientation histogram.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void smoothOrientationHist() const;

	///=================================================================================================
	/// <summary>
	/// Finds the dominant orientation of the histogram.
	/// </summary>
	/// 
	/// <returns> The dominant orientation. </returns>
	///-------------------------------------------------------------------------------------------------
	float findDominantOrientation() const;
	
	///=================================================================================================
	/// <summary>
	/// Resets the orientation histogram entries to 0.f.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void resetOrientationHist() const;
		
/*******************************************************************************************************
 ****** III.b (4) Feature Calculation ******************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Calculates the key points and the descriptors of each candidate. Stores them in the lists
	/// keyPoints and descriptors.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void calcFeatures() const;

	///=================================================================================================
	/// <summary>
	/// Calculates the key point of the given candidate.
	/// </summary>
	/// 
	/// <param name="candidate"> The candidate. </param>
	/// <param name="keyPoint"> [in,out] The key point. </param>
	///-------------------------------------------------------------------------------------------------
	void calcKeyPoint(const Candidate &candidate, cv::KeyPoint &keyPoint) const;
	
/*******************************************************************************************************
 ****** III.b (5) Descriptor Calculation ***************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Calculates the descriptor, represented as matrix, of the given candidate.
	/// </summary>
	/// 
	/// <param name="candidate"> The candidate. </param>
	/// <param name="descriptor"> [in,out] The descriptor. </param>
	///-------------------------------------------------------------------------------------------------
	void calcDescriptorMat(const Candidate &candidate, cv::Mat &descriptor) const;
		
	///=================================================================================================
	/// <summary>
	/// <para>Calculates the descriptor histogram of the given candidate.</para>
	/// <para>Multiple orientation histograms are created. The gradients to be put into the histograms
	/// are normalized to the candidate's orientation. The gradient samples are distributed to adjacent
	/// histograms and orientation bins. Afterwards the descriptor is normalized and trimmed.</para>
	/// </summary>
	/// 
	/// <param name="candidate"> [in,out] The candidate. </param>
	///-------------------------------------------------------------------------------------------------
	void calcDescriptor(const Candidate &candidate) const;

	///=================================================================================================
	/// <summary>
	/// Converts the current descriptor into a 1xN matrix where N is the DESCRIPTOR_LENGTH. The
	/// descriptor elements are multiplied by 512.f and limited to 255.
	/// </summary>
	/// 
	/// <param name="descriptor"> [in,out] The descriptor matrix. </param>
	///-------------------------------------------------------------------------------------------------
	void convertDescriptor2Mat(cv::Mat &descriptor) const;

	///=================================================================================================
	/// <summary>
	/// Distributes a gradient sample to adjacent histograms and orientation bins.
	/// </summary>
	/// 
	/// <param name="dcHist"> The horizontal sub histogram position. </param>
	/// <param name="drHist"> The vertical sub histogram position. </param>
	/// <param name="oBin"> The orientation bin. </param>
	/// <param name="magnitude"> The gradient's magnitude. </param>
	///-------------------------------------------------------------------------------------------------
	void distributeDescriptorEntry(float dcHist, float drHist, float oBin, float magnitude) const;
	
	///=================================================================================================
	/// <summary>
	/// Normalizes descriptor the descriptor to unit length.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void normalizeDescriptor() const;

	///=================================================================================================
	/// <summary>
	/// Trims the descriptor magnitudes. After trimming no magnitude is greater than the
	/// descriptorMagnitudeThreshold.
	/// </summary>
	/// 
	/// <returns> true if the descriptor needed to be trimmed. </returns>
	///-------------------------------------------------------------------------------------------------
	bool trimDescriptor() const;

	///=================================================================================================
	/// <summary>
	/// Normalizes and trims the descriptor. If needed the descriptor is renormalized afterwards.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void normalizeAndTrimDescriptor() const;

	///=================================================================================================
	/// <summary>
	/// Resets the descriptor entries to 0.f.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void resetDescriptor() const;

/******************************************************************************************************* 
 ****** III.c Math Helpers operating on current interval ***********************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Calculates the gradient of the current scale space interval.
	/// </summary>
	/// 
	/// <param name="c"> The pixel column. </param>
	/// <param name="r"> The pixel row. </param>
	/// <param name="magnitude"> [out] The gradient's magnitude. </param>
	/// <param name="angle"> [out] The gradient's angle. </param>
	/// 
	/// <returns> true if the gradient is available. </returns>
	///-------------------------------------------------------------------------------------------------
	bool calcGradient(int c, int r, float &magnitude, float &angle) const;

	///=================================================================================================
	/// <summary>
	/// Derives the given point (c, r) of the current dog interval in three dimension(row, column,
	/// scale).
	/// </summary>
	/// 
	/// <param name="c"> The pixel column. </param>
	/// <param name="r"> The pixel row. </param>
	/// 
	/// <returns> the derivation as 3x1 matrix: [dc dr ds]^T </returns>
	///-------------------------------------------------------------------------------------------------
	cv::Mat derive3D(int c, int r) const;

	///=================================================================================================
	/// <summary>
	/// Calculates the three dimensional (row, column, scale) hessian matrix of the given point (c, r)
	/// of the current dog interval.
	/// </summary>
	/// 
	/// <param name="c"> The pixel column. </param>
	/// <param name="r"> The pixel row. </param>
	/// 
	/// <returns>
	/// the hessian as 3x3 matrix: [ [dxx dxy dxs]^T [dxy dyy dys]^T [dxs dys dss]^T ]
	/// </returns>
	///-------------------------------------------------------------------------------------------------
	cv::Mat calcHessian3D(int c, int r) const;

};

}


#endif // SIFT_DETECTOR_H