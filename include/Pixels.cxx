#pragma once
typedef unsigned char uchar;

/*
    Red Green Blue Pixel, will handle all common operations
*/
struct PixelRGB{
    uchar blue;
    uchar green;
    uchar red;

    PixelRGB& operator=( const PixelRGB &other ){
        red = other.red;
        green = other.green;
        blue = other.blue;
        return *this;
    }

    PixelRGB& operator=( uchar value ){
        red = value;
        green = value;
        blue = value;
        return *this;
    }

    PixelRGB operator+( const PixelRGB& other ){
        PixelRGB new_pixel;
        new_pixel.red = red + other.red > 255 ? 255 : red + other.red;
        new_pixel.green = green + other.green > 255 ? 255 : green + other.green;
        new_pixel.blue = blue + other.blue > 255 ? 255 : blue + other.blue;
        return new_pixel;
    }

    PixelRGB operator-( const PixelRGB& other ){
        PixelRGB new_pixel;
        new_pixel.red = red - other.red < 0 ? 0 : red - other.red;
        new_pixel.green = green - other.green < 0 ? 0 : green - other.green;
        new_pixel.blue = blue - other.blue < 0 ? 0 : blue - other.blue;
        return new_pixel;
    }
};

/*
    Black and White Pixel, will handle all common operations
*/
struct PixelBW{
    uchar intensity;
    PixelBW& operator=( const PixelBW &other ){
        intensity = other.intensity;
        return *this;
    }

    PixelBW& operator=( uchar value ){
        intensity = value;
        return *this;
    }

    PixelBW operator+( const PixelBW& other ){
        PixelBW new_pixel;
        new_pixel.intensity = intensity + other.intensity > 255 ? 255 : intensity + other.intensity;
        return new_pixel;
    }

    PixelBW operator-( const PixelBW& other ){
        PixelBW new_pixel;
        new_pixel.intensity = intensity - other.intensity < 0 ? 0 : intensity - other.intensity;
        return new_pixel;
    }
};