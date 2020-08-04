#include <Image.h>

#ifdef CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif


/*
    This class will handle the raw data from the image
    taken from opencv, focused for cuda processig
*/

//Constructor, will reserve the memory needed for the program to run
template<typename Px>
Image<Px>::Image( cv::Mat input ){
    this->channels = input.channels();
    this->rows = input.rows;
    this->cols = input.cols;
    #ifdef CUDA
        cudaMallocManaged( &(this->pixels), sizeof(Px) * this->rows * this->cols );
    #else
        this->pixels = (Px *) malloc(sizeof(Px) * this->rows * this->cols );
    #endif
}


//Destructor, release memory when out of scope or when deleted
template<typename Px>
Image<Px>::~Image(){
    if( this->pixels != NULL ){
        #ifdef CUDA
            cudaFree(this->pixels);
        #else
            free(this->pixels);
        #endif
    }
}


//This operator will convert a CV Mat image into a Image class handled by CUDA(If its the case)
template<typename Px>
void operator >> ( cv::Mat &input, Image<Px> &img ){

    //Update number of channels
    img.channels = input.channels();

    //Copy raw pixels to memory
    #ifdef CUDA
        cudaMemcpy( img.pixels, input.ptr<uchar>(0), img.total, cudaMemcpyHostToDevice );
    #else
        memcpy( img.pixels, input.ptr<uchar>(0), img.total );
    #endif

    return;
}


template<typename Px>
//This method will return a CV::Mat, will convert the raw pixels into a compatible Mat structure
cv::Mat Image<Px>::cv_mat(){
    //Get channel type
    int type = this->channels == 1 ? CV_8UC1 : this->channels == 4 ? CV_8UC4 : this->channels == 3 ? CV_8UC3 : 0;

    //Create image from raw pixels
    cv::Mat img(this->rows, this->cols, type, (uchar*)this->pixels);

    return img;
}


