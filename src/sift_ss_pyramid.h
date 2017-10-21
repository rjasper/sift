#ifndef SIFT_SS_PYRAMID_H
#define SIFT_SS_PYRAMID_H


#include <opencv2/opencv.hpp>

#include "sift.h"


namespace sift {

///=================================================================================================
/// <summary>
/// The Scale Space Pyramid constains multiple interval images. It is constructed from an input
/// image which is successively blurred. Each octave begins with a halfed interval image from the
/// last octave. The pyramid is not build at once but in multiple steps. Therefore it is necessary
/// to call step to proceed to the next interval. You can access the previous and next interval
/// images which allows access the -1st and the (n + 2)th interval of an octave.
/// </summary>
///
/// <remarks> Jasper, 20.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class SsPyramid {

public:
	
/******************************************************************************************************* 
 ****** 0. Constructors ********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Default constructor. Constructs an empty object.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	SsPyramid();

	///=================================================================================================
	/// <summary>
	/// Constructor.
	/// </summary>
	///
	/// <param name="img"> The input image. </param>
	/// <param name="intervals"> (optional) The number of sampled intervals per octave. </param>
	/// <param name="sigma"> (optional) The standard deviation of each interval blurring. </param>
	/// <param name="sigmaInit">
	/// (optional) The assumed standard deviation of the initial input image blurring.
	/// </param>
	/// <param name="upscale">
	/// (optional) If set to true an additional octave "-1" is calculated. Width and height are doubled.
	/// If set to false the calculations begin in the 0th octave with original width and height.
	/// </param>
	///-------------------------------------------------------------------------------------------------
	SsPyramid(
		const cv::Mat &img,
		      bool    upscale   = UPSCALE,
		      int     intervals = INTERVALS,
		      float   sigma     = SIGMA,
		      float   sigmaInit = SIGMA_INIT);

/******************************************************************************************************* 
 ****** I. Public Methods ******************************************************************************
 ****** I.a Getters       ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary> Query if this object is empty. </summary>
	///
	/// <returns> true if empty, false if not. </returns>
	///-------------------------------------------------------------------------------------------------
	bool isEmpty() const;

	///=================================================================================================
	/// <summary>
	/// Gets the number of sampled intervals per octave.
	/// </summary>
	///
	/// <returns> The intervals. </returns>
	///-------------------------------------------------------------------------------------------------
	int getIntervals() const;

	///=================================================================================================
	/// <summary>
	/// Gets the current octave.
	/// </summary>
	///
	/// <returns> The octave. </returns>
	///-------------------------------------------------------------------------------------------------
	int getOctave() const;

	///=================================================================================================
	/// <summary>
	/// Gets the current interval.
	/// </summary>
	///
	/// <returns> The interval. </returns>
	///-------------------------------------------------------------------------------------------------
	int getInterval() const;
	
	///=================================================================================================
	/// <summary>
	/// Gets the previous interval image.
	/// </summary>
	///
	/// <returns> The previous interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getPrevious() const;
	
	///=================================================================================================
	/// <summary>
	/// Gets the current interval image.
	/// </summary>
	///
	/// <returns> The current interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getCurrent() const;

	///=================================================================================================
	/// <summary>
	/// Gets the next interval image.
	/// </summary>
	///
	/// <returns> The next interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getNext() const;
			
/******************************************************************************************************* 
 ****** I.b Use Case Methods ***************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary> Starts a new pyramid. </summary>
	///
	/// <param name="img"> The input image. </param>
	/// <param name="upscale"> (optional) If set to true an additional octave "-1" is calculated.
	/// 	Width and height are doubled. If set to false the calculations begin in the 0th octave
	/// 	with original width and height. </param>
	/// <param name="intervals"> (optional) The number of sampled intervals per octave. </param>
	/// <param name="sigma"> (optional) The standard deviation of each interval blurring. </param>
	/// <param name="sigmaInit"> (optional) The assumed standard deviation of the initial input image
	/// 	blurring. </param>
	///-------------------------------------------------------------------------------------------------
	void start(
		const cv::Mat &img,
		      bool    upscale   = UPSCALE,
		      int     intervals = INTERVALS,
		      float   sigma     = SIGMA,
		      float   sigmaInit = SIGMA_INIT);

	///=================================================================================================
	/// <summary>
	/// Calculates the next interval if the octave isn't fully build yet. Otherwise it builds the base
	/// of the next octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void step();
	
	///=================================================================================================
	/// <summary> Clears this object to its blank/initial state. </summary>
	///-------------------------------------------------------------------------------------------------
	void clear();

private:
		
/******************************************************************************************************* 
 ****** II. Fields    **********************************************************************************
 ****** II.a Settings **********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The number of sampled intervals per octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int intervals;

	///=================================================================================================
	/// <summary>
	/// The scale of the first octave which is doubled each following octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float sigma;
		
/******************************************************************************************************* 
 ****** II.b State *************************************************************************************
 *******************************************************************************************************/
	
	///=================================================================================================
	/// <summary>
	/// The current octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int octave;

	///=================================================================================================
	/// <summary>
	/// The current interval.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	int interval;

	///=================================================================================================
	/// <summary>
	/// The previous interval image.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	cv::Mat prev;

	///=================================================================================================
	/// <summary>
	/// The current interval image.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	cv::Mat curr;

	///=================================================================================================
	/// <summary>
	/// The next interval image.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	cv::Mat next;

	///=================================================================================================
	/// <summary>
	/// The factor to be applied to dsigma each interval.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float k;

	///=================================================================================================
	/// <summary>
	/// The standard deviation of the gaussian blurring which produces the next interval image by
	/// applying itself to the current interval image.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float dsigma;

	///=================================================================================================
	/// <summary>
	/// The standard deviation difference between the current octave's interval 0 and 1.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	float sigma1;
		
/******************************************************************************************************* 
 ****** III. Private Methodes **************************************************************************
 ****** III.a Initializer     **************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Initializes this object.
	/// </summary>
	///
	/// <param name="img"> The input image. </param>
	/// <param name="upscale">
	/// If set to true an additional octave "-1" is calculated. Width and height are doubled.
	/// If set to false the calculations begin in the 0th octave with original width and height.
	/// </param>
	/// <param name="sigmaInit">
	/// The assumed standard deviation of the initial input image blurring.
	/// </param>
	///-------------------------------------------------------------------------------------------------
	void init(const cv::Mat &img, bool upscale, float sigmaInit);

	///=================================================================================================
	/// <summary> Checks the input arguments. May throws domain errors. </summary>
	///
	/// <param name="intervals"> The number of sampled intervals per octave. </param>
	/// <param name="sigma"> The standard deviation of each interval blurring. </param>
	/// <param name="sigmaInit"> The assumed standard deviation of the initial input image
	/// 	blurring. </param>
	///-------------------------------------------------------------------------------------------------
	void check(int intervals, float sigma, float sigmaInit) const;
	
/******************************************************************************************************* 
 ****** III.b Pyramid Building *************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Builds the base of the current octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void buildBase();

	///=================================================================================================
	/// <summary>
	/// Builds the next interval.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void buildNext();
	
/******************************************************************************************************* 
 ****** III.c Helpers **********************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Downsamples the given matrix.
	/// </summary>
	///
	/// <param name="mat"> [in,out] The matrix. </param>
	///-------------------------------------------------------------------------------------------------
	void downsample(cv::Mat &mat) const;

};


/******************************************************************************************************* 
 ****** Inline Definitions *****************************************************************************
 *******************************************************************************************************/

inline bool SsPyramid::isEmpty() const {
	return curr.data == nullptr;
}

inline int SsPyramid::getIntervals() const {
	return intervals;
}

inline int SsPyramid::getOctave() const {
	return octave;
}

inline int SsPyramid::getInterval() const {
	return interval;
}

inline const cv::Mat &SsPyramid::getNext() const {
	return next;
}

inline const cv::Mat &SsPyramid::getCurrent() const {
	return curr;
}

inline const cv::Mat &SsPyramid::getPrevious() const {
	return prev;
}

}


#endif // SIFT_SS_PYRAMID_H