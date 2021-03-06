#ifndef SIFT_H
#define SIFT_H


namespace sift {


///=================================================================================================
/// <summary> The double image size before pyramid construction? </summary>
///-------------------------------------------------------------------------------------------------
const bool UPSCALE = true;

///=================================================================================================
/// <summary> The minimum amount of intervals allowed per octave. </summary>
///-------------------------------------------------------------------------------------------------
const int INTERVALS_MIN = 1;

///=================================================================================================
/// <summary> The default number of sampled intervals per octave. </summary>
///-------------------------------------------------------------------------------------------------
const int INTERVALS = 3;

///=================================================================================================
/// <summary> The resize quotient of each downsampling per octave </summary>
///-------------------------------------------------------------------------------------------------
const int OCTAVE_DOWNSAMPLING_DIVISOR = 2;

///=================================================================================================
/// <summary> The default sigma for initial gaussian smoothing. </summary>
///-------------------------------------------------------------------------------------------------
const float SIGMA = 1.6f;

///=================================================================================================
/// <summary> The assumed gaussian blur for input image. </summary>
///-------------------------------------------------------------------------------------------------
const float SIGMA_INIT = .5f;

///=================================================================================================
/// <summary> The maximum steps of keypoint interpolation before failure. </summary>
///-------------------------------------------------------------------------------------------------
const int MAX_INTERPOLATION_STEPS = 5;

///=================================================================================================
/// <summary> The default threshold on keypoint contrast |D(x)|. </summary>
///-------------------------------------------------------------------------------------------------
const float CONTRAST_THRESHOLD = .04f;

///=================================================================================================
/// <summary> The default threshold on keypoint ratio of principle curvatures. </summary>
///-------------------------------------------------------------------------------------------------
const float CURVATURE_THRESHOLD = 10.f;

///=================================================================================================
/// <summary> The width of border in which to ignore keypoints. </summary>
///-------------------------------------------------------------------------------------------------
const int IMAGE_BORDER = 5;

///=================================================================================================
/// <summary> The default number of bins in histogram for orientation assignment. </summary>
///-------------------------------------------------------------------------------------------------
const int ORIENTATION_BINS = 36;

///=================================================================================================
/// <summary> Determines gaussian sigma for orientation assignment. </summary>
///-------------------------------------------------------------------------------------------------
const float ORIENTATION_SIGMA_FACTOR = 1.5f;

///=================================================================================================
/// <summary> Determines the radius of the region used in orientation assignment. </summary>
///-------------------------------------------------------------------------------------------------
const float ORIENTATION_RADIUS_FACTOR = 3.f * ORIENTATION_SIGMA_FACTOR;

///=================================================================================================
/// <summary> The orientation magnitude relative to max that results in new feature. </summary>
///-------------------------------------------------------------------------------------------------
const float ORIENTATION_PEAK_RATIO = .8f;

///=================================================================================================
/// <summary> The number of iterations of orientation histogram smoothing. </summary>
///-------------------------------------------------------------------------------------------------
const int ORIENTATION_SMOOTH_ITERATIONS = 2;

///=================================================================================================
/// <summary> The candidate offset xc index. </summary>
///-------------------------------------------------------------------------------------------------
const int CANDIDATE_OFFSET_XC = 0;

///=================================================================================================
/// <summary> The candidate offset xr index. </summary>
///-------------------------------------------------------------------------------------------------
const int CANDIDATE_OFFSET_XR = 1;

///=================================================================================================
/// <summary> The candidate offset xi index. </summary>
///-------------------------------------------------------------------------------------------------
const int CANDIDATE_OFFSET_XI = 2;

///=================================================================================================
/// <summary> The default width of a descriptor histogram array. </summary>
///-------------------------------------------------------------------------------------------------
const int DESCRIPTOR_HIST_WIDTH = 4;

///=================================================================================================
/// <summary> The default number of bins per histogram in a descriptor array. </summary>
///-------------------------------------------------------------------------------------------------
const int DESCRIPTOR_HIST_BINS = 8;

///=================================================================================================
/// <summary> The dimension of the descriptor. </summary>
///-------------------------------------------------------------------------------------------------
const int DESCRIPTOR_LENGTH = (DESCRIPTOR_HIST_BINS * DESCRIPTOR_HIST_WIDTH * DESCRIPTOR_HIST_WIDTH);

///=================================================================================================
/// <summary> Determines the size of a single descriptor orientation histogram. </summary>
///-------------------------------------------------------------------------------------------------
const float DESCRIPTOR_SIGMA_FACTOR = 3.f;

///=================================================================================================
/// <summary> The threshold on magnitude of elements of descriptor vector. </summary>
///-------------------------------------------------------------------------------------------------
const float DESCRIPTOR_MAGNITUDE_THRESHOLD = .2f;

///=================================================================================================
/// <summary> The factor used to convert floating-point descriptor to unsigned char. </summary>
///-------------------------------------------------------------------------------------------------
const float DESCRIPTOR_FLOAT2UCHAR_FACTOR = 512.f;

///=================================================================================================
/// <summary> The amount of neighbors to be searched for matching purposes. </summary>
///-------------------------------------------------------------------------------------------------
const int K_NEAREST_NEIGHBORS = 2;

///=================================================================================================
/// <summary> The maximum distance ratio to the nearest neighbor used for matching. </summary>
///-------------------------------------------------------------------------------------------------
const float MATCH_NN_DISTANCE_RATIO = .8f;


}


#endif // SIFT_H