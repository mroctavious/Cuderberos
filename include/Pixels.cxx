#pragma once
#define BACKGROUND_BUFFER_SIZE 50

#ifdef CUDA
    #define HYBRID __host__ __device__
#else
    #define HYBRID
#endif

typedef unsigned char uchar;

/*
    Red Green Blue Pixel, will handle all common operations
*/
struct PixelRGB{
    uchar blue;
    uchar green;
    uchar red;

    HYBRID inline unsigned char CancholasBackgroundFormula( unsigned char fondo, unsigned char k, unsigned char new_pixel ){
        return (unsigned char)((((int)fondo * ((int)k - 1)) + (int)new_pixel) / (int)k);
    }

    HYBRID PixelRGB& operator=( const PixelRGB &other ){
        red = other.red;
        green = other.green;
        blue = other.blue;
        return *this;
    }

    HYBRID PixelRGB& operator=( uchar value ){
        red = value;
        green = value;
        blue = value;
        return *this;
    }

    HYBRID PixelRGB operator+( const PixelRGB& other ){
        PixelRGB new_pixel;
        new_pixel.red = red + other.red > 255 ? 255 : red + other.red;
        new_pixel.green = green + other.green > 255 ? 255 : green + other.green;
        new_pixel.blue = blue + other.blue > 255 ? 255 : blue + other.blue;
        return new_pixel;
    }

    HYBRID PixelRGB operator-( const PixelRGB& other ){
        PixelRGB new_pixel;
        new_pixel.red = red - other.red < 0 ? 0 : red - other.red;
        new_pixel.green = green - other.green < 0 ? 0 : green - other.green;
        new_pixel.blue = blue - other.blue < 0 ? 0 : blue - other.blue;
        return new_pixel;
    }
    
    HYBRID PixelRGB operator/( const PixelRGB &other){
        PixelRGB new_pixel;
        new_pixel.red = CancholasBackgroundFormula( red, BACKGROUND_BUFFER_SIZE, other.red );
        new_pixel.green = CancholasBackgroundFormula( green, BACKGROUND_BUFFER_SIZE, other.green );
        new_pixel.blue = CancholasBackgroundFormula( blue, BACKGROUND_BUFFER_SIZE, other.blue );
        return new_pixel;
    }
};

/*
    Black and White Pixel, will handle all common operations
*/
struct PixelBW{
    uchar intensity;
    HYBRID PixelBW& operator=( const PixelBW &other ){
        intensity = other.intensity;
        return *this;
    }

    HYBRID PixelBW& operator=( uchar value ){
        intensity = value;
        return *this;
    }

    HYBRID PixelBW operator+( const PixelBW& other ){
        PixelBW new_pixel;
        new_pixel.intensity = intensity + other.intensity > 255 ? 255 : intensity + other.intensity;
        return new_pixel;
    }

    HYBRID PixelBW operator-( const PixelBW& other ){
        PixelBW new_pixel;
        new_pixel.intensity = intensity - other.intensity < 0 ? 0 : intensity - other.intensity;
        return new_pixel;
    }
};