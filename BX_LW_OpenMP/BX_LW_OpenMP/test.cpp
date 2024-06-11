#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>

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
    int threads = 4;  
    omp_set_num_threads(threads);
    std::ifstream file("D:/MyVS/BX_LW/ExpIndex", std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    file.seekg(32832, std::ios::beg);
    std::vector<uint32_t> array1 = read_array(file);
    file.seekg(1733008, std::ios::beg);
    std::vector<uint32_t> array2 = read_array(file);
    file.close();
   /* std::ifstream file("D:/MyVS/BX_LW/ExpIndex", std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    file.seekg(32832, std::ios::beg);
    std::vector<uint32_t> array1 = read_array(file);
    std::vector<uint32_t> array2 = read_array(file);
    file.close();
    */
    std::vector<uint32_t> small_array;
    std::vector<uint32_t> large_array;
    if (array1.size() > array2.size()) {
        small_array=array2;
        large_array=array1;
    }
    else {
        small_array = array1;
        large_array = array2;
    } 

    std::vector<std::vector<uint32_t>> results;

    auto beforeTime = std::chrono::steady_clock::now();

    int num_threads;
   
#pragma omp parallel
    {
#pragma omp single
        {
            num_threads = omp_get_num_threads();
            results.resize(num_threads);
        }

        int thread_id = omp_get_thread_num();
        int total_elements = large_array.size();
        int chunk_size = total_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? total_elements : start + chunk_size;

        std::set_intersection(small_array.begin(), small_array.end(),
            large_array.begin() + start, large_array.begin() + end,
            std::back_inserter(results[thread_id]));
    }

    std::vector<uint32_t> final_result;
    for (const auto& vec : results) {
        final_result.insert(final_result.end(), vec.begin(), vec.end());
    }

    auto afterTime = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(afterTime - beforeTime).count();
   /* std::ofstream f3("D:/MyVS/BX_LW_OpenMP/result.txt", std::ios::app);
    if (!f3.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 0;
    }

    for (uint32_t value : final_result) {
        f3 << value << ' ';
    }
    f3.close();*/
    std::cout << "Intersection size: " << final_result.size() << ", time: " << time << " seconds" <<", num_thread: " <<num_threads<< std::endl;

    return 0;
}
