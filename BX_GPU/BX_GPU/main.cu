#include <iostream> 
#include <vector>    
#include <fstream>  
#include <chrono>    
#include <algorithm> 
#include <thrust/host_vector.h>  
#include <thrust/device_vector.h> 
#include <thrust/set_operations.h> 
#include <device_launch_parameters.h> 
#include <cuda.h>             
#include <stdio.h>         
#include <cuda_runtime.h>   

using namespace std;

#define THREADS_PER_BLOCK 256 // 每个CUDA块的线程数量定义

__device__ uint32_t* lower_bound(uint32_t* start, uint32_t* end, uint32_t value) {
    uint32_t* ptr;
    int count, step;
    count = end - start;

    while (count > 0) {
        ptr = start;
        step = count / 2;
        ptr += step;
        if (*ptr < value) {
            start = ++ptr;
            count -= step + 1;
        }
        else
            count = step;
    }
    return start;
}
__global__ void set_intersection(uint32_t* d_a, uint32_t* d_b, uint32_t* d_result, int n, int m) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        uint32_t value = d_a[i];
        uint32_t* found = lower_bound(d_b, d_b + m, value);
        if (found != (d_b + m) && *found == value) {
            d_result[i] = value;
        }
        else {
            d_result[i] = 0; // 设置为特殊值
        }
    }
}
std::vector<uint32_t> my_set_intersection(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    int n = a.size();
    int m = b.size();

    uint32_t* d_a;
    uint32_t* d_b;
    uint32_t* d_result;

    cudaMalloc(&d_a, n * sizeof(uint32_t));
    cudaMalloc(&d_b, m * sizeof(uint32_t));
    cudaMalloc(&d_result, n * sizeof(uint32_t));

    cudaMemcpy(d_a, a.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), m * sizeof(uint32_t), cudaMemcpyHostToDevice);


    set_intersection <<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (d_a, d_b, d_result, n, m);

    std::vector<uint32_t> result(n);
    cudaMemcpy(result.data(), d_result, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}
uint32_t read_uint32_le(std::ifstream& stream) {
    uint32_t value;
    char bytes[4];
    stream.read(bytes, 4);
    value = (static_cast<uint32_t>(static_cast<unsigned char>(bytes[3])) << 24) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[2])) << 16) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[1])) << 8) |
        static_cast<uint32_t>(static_cast<unsigned char>(bytes[0]));
    return value;
}

std::vector<uint32_t> read_array(std::ifstream& stream) {
    uint32_t length = read_uint32_le(stream);
    std::vector<uint32_t> array(length);
    for (uint32_t i = 0; i < length; ++i) {
        array[i] = read_uint32_le(stream);
    }
    return array;
}

int main() {
   /* std::ifstream file("D:/MyVS/ExpIndex", std::ios::binary);
    if (!file) {
        std::cout << "无法打开索引文件" << std::endl;
        return 1;
    }
    
    file.seekg(32832, std::ios::beg);  
    vector<uint32_t> array1 = read_array(file);
    vector<uint32_t> array2 = read_array(file);
    file.close();*/
    std::ifstream file("D:/MyVS/ExpIndex", std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    file.seekg(32832, std::ios::beg);
    vector<uint32_t> array1 = read_array(file);
    file.seekg(1733008, std::ios::beg);
    vector<uint32_t> array2 = read_array(file);
    file.close();

    thrust::device_vector<uint32_t> d_array1 = array1;
    thrust::device_vector<uint32_t> d_array2 = array2;

   
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> result = my_set_intersection(array1, array2);

    auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    
    std::ofstream f3("D:/MyVS/BX_GPU/result2.txt", std::ios::app);
    if (!f3.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 0;
    }
    int i = 0;
    for (uint32_t value : result) {
        if (value != 0) {
        f3 << value << ' ';
        i++;
    }
    }
    f3.close();
   
    std::cout << "运行时间" << time << "微秒" << " ,size:"<<i<<endl;
   
}
