#include <hip/hip_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#define CHANNELS 3


__global__ void grayfy(unsigned char *Pout, unsigned char *Pin, int row_len, int col_len){
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (col < col_len && row < row_len){
		int grayidx = row*col_len + col;

		int rgbidx = grayidx*CHANNELS;
		unsigned char r = Pin[rgbidx], g = Pin[rgbidx+1], b=Pin[rgbidx+2];
		
		Pout[grayidx] = 0.21f*r+0.71f*g+0.07f*b;
	}
}

void gray(const unsigned char *Pin, unsigned char *Pout, int row_len, int col_len){
	unsigned char *Pin_d, *Pout_d;
	std::size_t size = static_cast<std::size_t>(row_len) * col_len * sizeof(unsigned char);

	dim3 dimGrid(ceil(col_len/16.0), ceil(row_len/16.0),1);
	dim3 dimBlock(16,16,1);


	hipMalloc((void**) &Pin_d, size*CHANNELS);
	hipMalloc((void**) &Pout_d, size);

	hipMemcpy(Pin_d, Pin, size*CHANNELS, hipMemcpyHostToDevice);

	grayfy<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, row_len, col_len);
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
    cv::Mat gray_img(img.rows, img.cols, CV_8UC1);

    // Run HIP grayscale conversion
    gray(data, gray_img.data, img.cols, img.rows);

    // Build output filename (e.g., gray_input.png)
    std::string out_name = "gray_" + filename;

    if (!cv::imwrite(out_name, gray_img)) {
        std::cerr << "Failed to save grayscale image: " << out_name << "\n";
        return 1;
    }

    std::cout << "Saved grayscale image to: " << out_name << "\n";
    return 0;
}

