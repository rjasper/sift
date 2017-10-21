#ifndef SIFT_EXCEPTIONS_H
#define SIFT_EXCEPTIONS_H


#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>


namespace sift {

///=================================================================================================
/// <summary>
/// Pyramid empty error. Is thrown if the pyramid is empty.
/// </summary>
///
/// <remarks> Jasper, 24.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class PyramidEmptyError : public std::logic_error {

public:

	PyramidEmptyError(const std::string &msg);

};

///=================================================================================================
/// <summary>
/// Pyramid downsampling error. Is thrown if the pyramid cannot perform any more downsamplings.
/// </summary>
///
/// <remarks> Jasper, 21.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class PyramidDownsamplingError : public std::logic_error {

public:

	///=================================================================================================
	/// <summary> Constructor. </summary>
	///
	/// <param name="msg"> The error message. </param>
	/// <param name="octave"> The octave which couldn't be downsampled. </param>
	///-------------------------------------------------------------------------------------------------
	PyramidDownsamplingError(const std::string &msg, int octave);

	///=================================================================================================
	/// <summary> The octave which couldn't be downsampled. </summary>
	///-------------------------------------------------------------------------------------------------
	const int octave;

};

///=================================================================================================
/// <summary>
/// Bad image format. Is thrown when an image cannot be processed due its format.
/// </summary>
///
/// <remarks> Jasper, 21.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class BadImageFormat : public std::invalid_argument {

public:

	///=================================================================================================
	/// <summary> Constructor. </summary>
	///
	/// <param name="msg"> The message. </param>
	/// <param name="img"> The image which couldn't be processed due its format. </param>
	///-------------------------------------------------------------------------------------------------
	BadImageFormat(const std::string &msg, const cv::Mat &img);

	/// <summary> The image which couldn't be processed due its format. </summary>
	const cv::Mat img;

};

}


#endif // SIFT_EXCEPTIONS_H