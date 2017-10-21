#ifndef SIFT_IO_H
#define SIFT_IO_H


#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "sift_model.h"


namespace sift {

///=================================================================================================
/// <summary> Writes the features to an output stream. </summary>
///
/// <param name="out"> [in,out] The output stream. </param>
/// <param name="keyPoints"> The key points. </param>
/// <param name="descriptors"> The descriptors. </param>
///-------------------------------------------------------------------------------------------------
void writeFeatures(
	      std::ostream              &out,
	const std::vector<cv::KeyPoint> &keyPoints,
	const cv ::Mat                  &descriptors);

///=================================================================================================
/// <summary> Reads features from an input stream. </summary>
///
/// <param name="in"> [in,out] The input stream. </param>
/// <param name="keyPoints"> [in,out] The key points. </param>
/// <param name="descriptors"> [in,out] The descriptors. </param>
///-------------------------------------------------------------------------------------------------
void readFeatures(
	std::istream              &in,
	std::vector<cv::KeyPoint> &keyPoints,
	cv ::Mat                  &descriptors);

///=================================================================================================
/// <summary>
/// Writes a model to an output stream.
/// </summary>
///
/// <param name="out"> [in,out] The output stream. </param>
/// <param name="model"> The model. </param>
///
/// <returns> The output stream. </returns>
///-------------------------------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &out, const Model &model) {
	model.write(out);
	return out;
}

///=================================================================================================
/// <summary>
/// Extracts a model from an input stream.
/// </summary>
///
/// <param name="in"> [in,out] The input stream. </param>
/// <param name="model"> [in,out] The model to be loaded from the stream. </param>
///
/// <returns> The input stream. </returns>
///-------------------------------------------------------------------------------------------------
inline std::istream &operator>>(std::istream &in, Model &model) {
	model.read(in);
	return in;
}

}


#endif // SIFT_IO_H