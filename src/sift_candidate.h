#ifndef SIFT_CANDIDATE_H
#define SIFT_CANDIDATE_H


#include <opencv2/opencv.hpp>


namespace sift {

///=================================================================================================
/// <summary>
/// The Candidate stores data used for key point and descriptor extraction. It is used by the
/// Detector.
/// </summary>
///
/// <remarks> Jasper, 20.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
struct Candidate {

	/// <summary> The pixel column in the current interval. </summary>
	int c;
	/// <summary> The pixel row in the current interval. </summary>
	int r;

	/// <summary> The subpixel offset of the interpolated extremum. </summary>
	cv::Mat offset;

	/// <summary> The radius. </summary>
	float radius;

	/// <summary> Size of a descriptor sub histogram. </summary>
	float histSize;

	/// <summary> The candidate's scale relative to the current octave. </summary>
	float octaveScale;

	/// <summary> The orientation. </summary>
	float orientation;

};


}


#endif // SIFT_CANDIDATE_H