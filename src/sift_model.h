#ifndef SIFT_MODEL_H
#define SIFT_MODEL_H


#include <functional>

#include <vector>

#include <opencv2/opencv.hpp>


namespace sift {

class Detector;

///=================================================================================================
/// <summary>
/// The Model stores the key points and descriptors of an image object. You can compare a model to
/// others for object recognition.
/// sift_io.h provides I/O functions for this class.
/// </summary>
///
/// <remarks> Jasper, 20.08.2012. </remarks>
///-------------------------------------------------------------------------------------------------
class Model {

public:
	
/******************************************************************************************************* 
 ****** 0. Constructors and Destructors ****************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Default constructor. Initializes an empty set of key points and descriptors.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	Model();

	///=================================================================================================
	/// <summary>
	/// Copy constructor.
	/// </summary>
	///
	/// <param name="other"> The model to be copied. </param>
	///-------------------------------------------------------------------------------------------------
	Model(const Model &other);

	///=================================================================================================
	/// <summary> Move constructor. </summary>
	///
	/// <param name="other"> The model to be moved. </param>
	///-------------------------------------------------------------------------------------------------
	Model(Model &&other);

	///=================================================================================================
	/// <summary>
	/// Constructs a model from the given set of key points and descriptors.
	/// </summary>
	///
	/// <param name="keyPoints"> The key points. </param>
	/// <param name="descriptors"> The descriptors. </param>
	///-------------------------------------------------------------------------------------------------
	Model(const std::vector<cv::KeyPoint> &keyPoints, const cv::Mat &descriptors);

	///=================================================================================================
	/// <summary>
	/// Constructs a model by moving the given set of key points and assigning descriptors.
	/// </summary>
	///
	/// <param name="keyPoints"> The key points. </param>
	/// <param name="descriptors"> The descriptors. </param>
	///-------------------------------------------------------------------------------------------------
	Model(const std::vector<cv::KeyPoint> &&keyPoints, const cv::Mat &descriptors);

	///=================================================================================================
	/// <summary>
	/// Destructs this object and the matcher if initialized.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	~Model();
		
/******************************************************************************************************* 
 ****** I. Public Methods ******************************************************************************
 ****** I.a Operators     ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Copies the given model. The copy contains a copied version of the key points and descriptors.
	/// The descriptors have a new matrix header which points to the same data. The matcher pointer is
	/// initialized with the null pointer.
	/// </summary>
	///
	/// <param name="rhs">
	/// The model to be copied.
	/// </param>
	///
	/// <returns> A copy of this object. </returns>
	///-------------------------------------------------------------------------------------------------
	Model &operator=(const Model &rhs);

	///=================================================================================================
	/// <summary>
	/// Moves the given model.
	/// </summary>
	///
	/// <param name="rhs"> [in,out] The model to be moved. </param>
	///
	/// <returns> A reference to this object </returns>
	///-------------------------------------------------------------------------------------------------
	Model &operator=(Model &&rhs);
		
/******************************************************************************************************* 
 ****** I.b Getters       ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Queries if this object is empty.
	/// </summary>
	///
	/// <returns> true if empty, false if not. </returns>
	///-------------------------------------------------------------------------------------------------
	bool isEmpty() const;

	///=================================================================================================
	/// <summary>
	/// Gets the key points.
	/// </summary>
	///
	/// <returns> The key points. </returns>
	///-------------------------------------------------------------------------------------------------
	const std::vector<cv::KeyPoint> &getKeyPoints() const;

	///=================================================================================================
	/// <summary>
	/// Gets the descriptors.
	/// </summary>
	///
	/// <returns> The descriptors. </returns>
	///-------------------------------------------------------------------------------------------------
	const cv::Mat &getDescriptors() const;

	///=================================================================================================
	/// <summary>
	/// Returns the amount of features.
	/// </summary>
	///
	/// <returns> The amount of features. </returns>
	///-------------------------------------------------------------------------------------------------
	int size() const;
			
/******************************************************************************************************* 
 ****** I.c Setters       ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary> Sets the features of this model. </summary>
	///
	/// <param name="keyPoints"> The key points of the features. </param>
	/// <param name="descriptors"> The descriptors of the features. </param>
	///-------------------------------------------------------------------------------------------------
	void setFeatures(const std::vector<cv::KeyPoint> &keyPoints, const cv::Mat &descriptors);
	
	///=================================================================================================
	/// <summary> Moves the key points and assigns the descriptors. </summary>
	///
	/// <param name="keyPoints"> The key points of the features. </param>
	/// <param name="descriptors"> The descriptors of the features. </param>
	///-------------------------------------------------------------------------------------------------
	void setFeatures(const std::vector<cv::KeyPoint> &&keyPoints, const cv::Mat &descriptors);

/******************************************************************************************************* 
 ****** I.d Use Case Methods ***************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Extracts the model from an input stream.
	/// </summary>
	///
	/// <param name="in"> [in,out] The input stream. </param>
	///
	/// <returns> The input stream. </returns>
	///-------------------------------------------------------------------------------------------------
	void read(std::istream &in);

	///=================================================================================================
	/// <summary>
	/// Writes the model to an output stream.
	/// </summary>
	///
	/// <param name="out"> [in,out] The output stream. </param>
	///
	/// <returns> The output stream. </returns>
	///-------------------------------------------------------------------------------------------------
	void write(std::ostream &out) const;

	///=================================================================================================
	/// <summary>
	/// Detects a model from the given image using the given detector.
	/// </summary>
	///
	/// <param name="detector"> The detector. </param>
	/// <param name="img"> The image. </param>
	///
	/// <returns> The number of found features. </returns>
	///-------------------------------------------------------------------------------------------------
	int detect(const sift::Detector &detector, const cv::Mat &img);

	///=================================================================================================
	/// <summary>
	/// Matches the given descriptor against this object.
	/// </summary>
	///
	/// <param name="descriptors"> The descriptors. </param>
	/// <param name="matches"> [in,out] The matches. </param>
	///
	/// <returns> The number of matches </returns>
	///-------------------------------------------------------------------------------------------------
	int match(const cv::Mat &descriptors, std::vector<cv::DMatch> &matches) const;

	///=================================================================================================
	/// <summary>
	/// Matches the given model against this object.
	/// </summary>
	///
	/// <param name="model"> The model. </param>
	/// <param name="matches"> [in,out] The matches. </param>
	///
	/// <returns> The number of matches </returns>
	///-------------------------------------------------------------------------------------------------
	int match(const Model &model, std::vector<cv::DMatch> &matches) const;

	///=================================================================================================
	/// <summary>
	/// Filters the given the key points by the given condition.
	/// </summary>
	///
	/// <param name="condition"> The condition. </param>
	///-------------------------------------------------------------------------------------------------
	void filter(const std::function<bool (const cv::KeyPoint &)> &condition);

	///=================================================================================================
	/// <summary>
	/// Maps the key points using the given mapper.
	/// </summary>
	///
	/// <param name="mapper"> The mapper. </param>
	///-------------------------------------------------------------------------------------------------
	void map(const std::function<void (cv::KeyPoint &)> &mapper);

	///=================================================================================================
	/// <summary>
	/// Clears this object to its blank/initial state.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void clear();

private:
		
/******************************************************************************************************* 
 ****** II Fields              *************************************************************************
 ****** II.a Actual Model Data *************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// The key points. The indices comply with the rows of descriptors.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	std::vector<cv::KeyPoint> keyPoints;

	///=================================================================================================
	/// <summary>
	/// The descriptors. The rows comply with the indices of keyPoints.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	cv::Mat descriptors;
		
/******************************************************************************************************* 
 ****** II.b Mutable Data ******************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Stores a trained matcher for later use. Is null if not trained yet.
	/// </summary>
	///
	/// <value> The matcher. </value>
	///-------------------------------------------------------------------------------------------------
	mutable cv::DescriptorMatcher *matcher;
	
/******************************************************************************************************* 
 ****** III. Private Methodes **************************************************************************
 ****** III.a Initializer     **************************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Initializes this object.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void init();
	
/******************************************************************************************************* 
 ****** III.b Miscellaneous Methods ********************************************************************
 *******************************************************************************************************/

	///=================================================================================================
	/// <summary>
	/// Trains the matcher.
	/// </summary>
	///-------------------------------------------------------------------------------------------------
	void train() const;

	///=================================================================================================
	/// <summary>
	/// Matches the given key points and descriptor against this object.
	/// </summary>
	///
	/// <param name="descriptors"> The descriptors. </param>
	/// <param name="matches"> [in,out] The matches. </param>
	///
	/// <returns> The number of matches </returns>
	///-------------------------------------------------------------------------------------------------
	int matchImpl(const cv::Mat &descriptors, std::vector<cv::DMatch> &matches) const;

	///=================================================================================================
	/// <summary>
	/// Checks the given features for consistency.
	/// </summary>
	///
	/// <param name="keyPoints"> The key points. </param>
	/// <param name="descriptors"> The descriptors. </param>
	///-------------------------------------------------------------------------------------------------
	void checkFeatures(const std::vector<cv::KeyPoint> &keyPoints, const cv::Mat &descriptors) const;

};


/******************************************************************************************************* 
 ****** Inline Definitions *****************************************************************************
 *******************************************************************************************************/

inline bool Model::isEmpty() const {
	return keyPoints.size() == 0;
}

inline const std::vector<cv::KeyPoint> &Model::getKeyPoints() const {
	return keyPoints;
}

inline const cv::Mat &Model::getDescriptors() const {
	return descriptors;
}

inline int Model::size() const {
	return keyPoints.size();
}

}


#endif // SIFT_MODEL_H