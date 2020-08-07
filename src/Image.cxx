#include <Image.h>

#ifdef CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename PixelType>
__global__ void apply_background_formula( PixelType *background, PixelType *new_frame){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    background[index]= background[index] / new_frame[index];
}

int get_threads( int n, int thrds ){
    return n % thrds == 0 ? thrds : get_threads(n, thrds/2);
}
#endif



/*
    This class will handle the raw data from the image
    taken from opencv, focused for cuda processig
*/

//Constructor, will reserve the memory needed for the program to run
template<typename PixelType>
Image<PixelType>::Image( cv::Mat &input ){
    this->channels = input.channels();
    this->rows = input.rows;
    this->cols = input.cols;
    this->total = this->rows * this->cols;
    #ifdef CUDA
        cudaMallocManaged( (void**)&(this->pixels), sizeof(PixelType) * this->rows * this->cols );
        this->threads = get_threads(this->total, 32);
        this->blocks = this->total / this->threads;
    #else
        this->pixels = (PixelType *) malloc(sizeof(PixelType) * this->rows * this->cols );
    #endif

    #ifdef DEBUG
        std::cout << "Rows: " << this->rows << std::endl;
        std::cout << "Cols: " << this->cols << std::endl;
        std::cout << "Total: " << this->total << std::endl;
        std::cout << "Channels: " << this->channels << std::endl;
        std::cout << "Data ptr: " << this->pixels << std::endl;

        #ifdef CUDA
            std::cout << "Thrds: " << this->threads << std::endl;
            std::cout << "Blocks: " << this->blocks << std::endl;
        #endif
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


//This method will convert a CV Mat image into a Image class handled by CUDA(If its the case)
template<typename Px>
void Image<Px>::set( cv::Mat &input){

    //Update number of channels
    this->channels = input.channels();

    //Copy raw pixels to memory
    #ifdef CUDA
        cudaMemcpy( this->pixels, input.ptr<uchar>(0), this->total*this->channels, cudaMemcpyHostToDevice );
    #else
        memcpy( this->pixels, input.ptr<uchar>(0), this->total*this->channels );
    #endif

    return;
}


template<typename Px>
//This method will return a CV::Mat, will convert the raw pixels into a compatible Mat structure
cv::Mat Image<Px>::cv_mat(){
    //Get channel type
    int type = this->channels == 1 ? CV_8UC1 : this->channels == 4 ? CV_8UC4 : this->channels == 3 ? CV_8UC3 : 0;

    #ifdef CUDA
        cudaDeviceSynchronize();
    #endif

    //Create image from raw pixels
    cv::Mat img(this->rows, this->cols, type, (uchar*)this->pixels);

    return img;
}

template<typename Px>
void Image<Px>::create_background( cv::VideoCapture &capture ){
    cv::Mat frame;
    size_t total_bytes = (size_t)this->total * sizeof(Px);
    int *background_pixels = (int *) malloc( sizeof(int) * total_bytes );

    #ifdef CUDA
        cudaDeviceSynchronize();
    #endif
    for( unsigned int j=0; j < total_bytes; j++ ){
        background_pixels[j] = 0;
    }

    for( unsigned int i=0; i < BACKGROUND_BUFFER_SIZE; i++ ){
        capture >> frame;
        if( frame.empty() ) break;

        unsigned char *raw = frame.ptr<unsigned char>(0);

        for( unsigned int j=0; j < total_bytes; j++ ){
            background_pixels[j] += raw[j];
        }
    }

    for( unsigned int j=0; j < this->total; j++ ){
        this->pixels[j].red = (uchar)(background_pixels[j*this->channels+2] / BACKGROUND_BUFFER_SIZE );
        this->pixels[j].green = (uchar)(background_pixels[j*this->channels+1] / BACKGROUND_BUFFER_SIZE );
        this->pixels[j].blue = (uchar)(background_pixels[j*this->channels] / BACKGROUND_BUFFER_SIZE );
    }

    free(background_pixels);
}

template<typename Px>
void Image<Px>::update_background( Image<Px> &other){
    #ifdef CUDA
        apply_background_formula<Px><<<this->blocks, this->threads>>>( this->pixels, other.pixels);
    #else
        for( unsigned int i=0; i<this->total; i++ ){
            this->pixels[i] = this->pixels[i] / other.pixels[i];
        }
    #endif
    return;
}

template class Image<PixelRGB>;