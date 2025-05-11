
#include <iostream>

int main(void) {

	int value;
	int device;
	
	cudaGetDevice(&device);
	cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, device);

	std::cout << "number of SMs: " << value << " on device: " << device << std::endl;
	return 0;
}