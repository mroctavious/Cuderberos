#pragma once
#include "Pixels.cxx"
#include <opencv2/opencv.hpp>
#include <string.h>
typedef unsigned char byte;

/*
    This class will handle the raw data from the image
    taken from opencv, focused for cuda processig
*/
template <typename PixelType>
class Image{
public:
    //Number of channels of the image
    int channels;

    //Number of rows and columns of the image
    unsigned int rows;
    unsigned int cols;

    #ifdef CUDA
        //Number of blocks and threads that can be used for the image
        unsigned int blocks;
        unsigned int threads;
    #endif

    //Total number of pixels
    unsigned int total;

    //Pointer to the raw pixels
    PixelType *pixels;

    //Constructor, will reserve the memory needed for the program to run
    Image( cv::Mat& );

    //Destructor, release memory when out of scope or when deleted
    ~Image();

    cv::Mat cv_mat();

    //This operator will convert a CV Mat image into a Image class handled by CUDA(If its the case)
    void set( cv::Mat &input );

    void create_background( cv::VideoCapture & );
    void update_background( Image & );

};
