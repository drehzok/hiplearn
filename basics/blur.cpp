#include <hip/hip_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#define CHANNELS 3
#define BLURSIZE 1


/*
This code is adapted from grayscale.cpp

Let's quickly remind what should be done (adaptation fixes)
 - [ ] hipmalloc bytes adjustment
 - [ ] 

Pseudocode for the blurring process
input: array (c,r,3) 
output: array (c,r,3)

for blurring let's take 3x3 neighborhood to blur

each thread will deal with (3,3) maximum block z-axis - channel

 */

__global__ void blurrify(unsigned char *Pout, unsigned char *Pin, int row_len, int col_len){
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
    int chan = blockIdx.z;
	if (col < col_len && row < row_len){
		int pixval = 0;
        int pixcount = 0;

		int idx = (row*col_len + col)*CHANNELS + chan;

        for (int ix = -BLURSIZE; ix<BLURSIZE+1; ++ix){
            for (int iy = -BLURSIZE; iy<BLURSIZE+1; ++iy){
                int currow = row + iy;
                int curcol = col + ix;
                if (currow >=0 && curcol >=0 && curcol < col_len && currow < row_len){
                    pixval += Pin[(currow*col_len + curcol)*CHANNELS + chan];
                    ++pixcount;
                }
            }
        }
		
		Pout[idx] = (unsigned char) (pixval/pixcount);
	}
}

void gblur(const unsigned char *Pin, unsigned char *Pout, int row_len, int col_len){
	unsigned char *Pin_d, *Pout_d;
	std::size_t size = static_cast<std::size_t>(row_len) * col_len * sizeof(unsigned char) * CHANNELS;

	dim3 dimGrid(ceil(col_len/16.0), ceil(row_len/16.0),CHANNELS);
	dim3 dimBlock(16,16,1);


	hipMalloc((void**) &Pin_d, size);
	hipMalloc((void**) &Pout_d, size);

	hipMemcpy(Pin_d, Pin, size, hipMemcpyHostToDevice);

	blurrify<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, row_len, col_len);
	hipMemcpy(Pout, Pout_d, size, hipMemcpyDeviceToHost);

	hipFree(Pin_d);
	hipFree(Pout_d);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " image_file\n";
        return 1;
    }

    std::string filename = argv[1];

    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << filename << "\n";
        return 1;
    }

    // Ensure contiguous data
    if (!img.isContinuous()) {
        img = img.clone();
    }

    std::cout << "Loaded " << filename << "\n";
    std::cout << "Size: " << img.cols << "x" << img.rows
              << ", channels: " << img.channels() << "\n";

    const unsigned char* data = img.data;
    std::size_t num_bytes =
        static_cast<std::size_t>(img.total()) * img.channels();
    std::cout << "Image buffer size: " << num_bytes << " bytes\n";

    // Prepare output grayscale Mat: 8-bit, single channel
    cv::Mat gray_img(img.rows, img.cols, CV_8UC3);

    // Run HIP grayscale conversion
    gblur(data, gray_img.data, img.rows, img.cols);

    // Build output filename (e.g., gray_input.png)
    std::string out_name = "gray_" + filename;

    if (!cv::imwrite(out_name, gray_img)) {
        std::cerr << "Failed to save grayscale image: " << out_name << "\n";
        return 1;
    }

    std::cout << "Saved grayscale image to: " << out_name << "\n";
    return 0;
}

