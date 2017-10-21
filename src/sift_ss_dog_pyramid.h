#ifndef SIFT_SS_DOG_PYRAMID_H
#define SIFT_SS_DOG_PYRAMID_H

#include <opencv2/opencv.hpp>

#include "sift_ss_pyramid.h"
#include "sift_dog_pyramid.h"


namespace sift {

///=================================================================================================
/// <summary>
/// The SS-DoG Pyramid combines a scale space and difference-of-Gaussian pyramid. This simplifies
/// the build of the dog pyramid since the pyramid steps are managed through this class.
/// </summary>
///
/// <remarks> Jasper, 20.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class SsDogPyramid {

public:
		
/******************************************************************************************************* 
 ****** 0. Constructors ********************************************************************************
 *******************************************************************************************************/
	
	///=================================================================================================
	/// <summary>
	/// Default constructor. Constructs an empty object which cannot be used.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	SsDogPyramid();

	///=================================================================================================
	/// <summary> Constructor. </summary>
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
	SsDogPyramid(
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
	/// <returns> The interval interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	int getInterval() const;

	///=================================================================================================
	/// <summary>
	/// Gets the current scale space interval image.
	/// </summary>
	///
	/// <returns> The current scale space interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getCurrentSs() const;

	///=================================================================================================
	/// <summary>
	/// Gets the previous dog interval image.
	/// </summary>
	///
	/// <returns> The previous dog interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getPreviousDog() const;

	///=================================================================================================
	/// <summary>
	/// Gets the current dog interval image.
	/// </summary>
	///
	/// <returns> The current dog interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getCurrentDog() const;
	
	///=================================================================================================
	/// <summary>
	/// Gets the next dog interval image.
	/// </summary>
	///
	/// <returns> The next dog interval image. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getNextDog() const;

	///=================================================================================================
	/// <summary>
	/// Gets the current interval image's size.
	/// </summary>
	///
	/// <returns> The current size. </returns>
	///-------------------------------------------------------------------------------------------------
	cv::Size getCurrentSize() const;
				
/******************************************************************************************************* 
 ****** I.b Use Case Methods ***************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary> Starts a new SS-DoG-Pyramid pair. </summary>
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
	/// <summary> Clears this object to its blank/initial state. </summary>
	///-------------------------------------------------------------------------------------------------
	void clear();

	///=================================================================================================
	/// <summary>
	/// Calculates the next interval if the octave isn't fully build yet. Otherwise it builds the base
	/// of the next octave.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void step();

private:
				
/******************************************************************************************************* 
 ****** II. Fields *************************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The Scale Space Pyramid.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	SsPyramid ss;

	///=================================================================================================
	/// <summary>
	/// The Difference-of-Gaussian Pyramid.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	DogPyramid dog;
			
/******************************************************************************************************* 
 ****** III. Private Methodes **************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Initializes this object.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void init();

};

inline bool SsDogPyramid::isEmpty() const {
	return ss.isEmpty();
}

inline int SsDogPyramid::getIntervals() const {
	return ss.getIntervals();
}

inline int SsDogPyramid::getInterval() const {
	return ss.getInterval() - 1;
}

inline int SsDogPyramid::getOctave() const {
	return ss.getOctave();
}
	
inline const cv::Mat &SsDogPyramid::getCurrentSs() const {
	return ss.getPrevious();
}

inline const cv::Mat &SsDogPyramid::getNextDog() const {
	return dog.getNext();
}
	
inline const cv::Mat &SsDogPyramid::getCurrentDog() const {
	return dog.getCurrent();
}
	
inline const cv::Mat &SsDogPyramid::getPreviousDog() const {
	return dog.getPrevious();
}

}


#endif // SIFT_SS_DOG_PYRAMID_H