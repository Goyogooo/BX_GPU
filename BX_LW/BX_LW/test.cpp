#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace std;
// 读取小端格式的四字节无符号整数
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

// 读取一个整数数组
std::vector<uint32_t> read_array(std::ifstream& stream) {
    uint32_t length = read_uint32_le(stream);
    std::vector<uint32_t> array(length);
    for (uint32_t i = 0; i < length; ++i) {
        array[i] = read_uint32_le(stream);
    }
    return array;
}

// Function to find the intersection of two sorted arrays
std::vector<uint32_t> findIntersection(const std::vector<uint32_t>& array1, const std::vector<uint32_t>& array2) {
    std::vector<uint32_t> result;
    // Iterate over the elements of the smaller array
    const std::vector<uint32_t>& smaller = array1.size() <= array2.size() ? array1 : array2;
    const std::vector<uint32_t>& larger = array1.size() > array2.size() ? array1 : array2;

    for (uint32_t element : smaller) {
        // Use binary search to check if the element exists in the larger array
        if (std::binary_search(larger.begin(), larger.end(), element)) {
            result.push_back(element);
        }
    }

    return result;
}

int main() {
    // double arg = 0;
   // for (int j = 0; j < 1000; j++) {
    std::ifstream file("D:/MyVS/BX_LW/ExpIndex", std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    file.seekg(32832, std::ios::beg);
    vector<uint32_t> array1 = read_array(file);
    file.seekg(1733008, std::ios::beg);
    vector<uint32_t> array2 = read_array(file);
   
    /*
        std::ifstream file("D:/MyVS/BX_LW/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件" << std::endl;
            return 1;
        }
        file.seekg(32832, std::ios::beg);
        std::vector<uint32_t> array1 = read_array(file);
        std::vector<uint32_t> array2 = read_array(file);
   */
        auto beforeTime = std::chrono::steady_clock::now();
        vector<uint32_t> result = findIntersection(array1, array2);
        cout << result.size();
        auto afterTime = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        std::cout << " time=" << time << "seconds" << " ,size:"<<result.size()<<std::endl;
       /* std::vector<float> floatArray;
           转换整数数组到浮点数组
        for (int value : array) {
            floatArray.push_back(static_cast<float>(value));
        }
        uint32_t length = floatArray.size();
        cout << length;
        */
       /* std::vector<float> compress(length);
        compress[0] = floatArray[0];
        //auto beforeTime = std::chrono::steady_clock::now();
        for (uint32_t i = 1; i < length; i++)
        {
            compress[i] = array[i] - array[i - 1];
        }
        */
        // auto afterTime = std::chrono::steady_clock::now();
         //double time = std::chrono::duration<double>(afterTime - beforeTime).count();
        // arg += time;
        // std::cout << " time=" << time << "seconds" << std::endl;
        /* std::ofstream f("D:/MyVS/BX_LW/array1.txt", std::ios::app);
         if (!f.is_open()) {
             std::cerr << "无法打开文件" << std::endl;
             return 0;
         }
         for (uint32_t value : array1) {
             f << value << ' ';
         }
         f.close();
         std::ofstream f2("D:/MyVS/BX_LW/array2.txt", std::ios::app);
         if (!f2.is_open()) {
             std::cerr << "无法打开文件" << std::endl;
             return 0;
         }
         for (uint32_t value : array2) {
             f2 << value << ' ';
         }
         f2.close();*/
     /*   std::ofstream f3("D:/MyVS/BX_LW/result.txt", std::ios::app);
         if (!f3.is_open()) {
             std::cerr << "无法打开文件" << std::endl;
             return 0;
         }
         
         for (uint32_t value : result) {
             f3 << value << ' ';
         }
         f3.close();
       */  
         file.close();
   // }
    // std::cout << " time=" << arg/100 << "seconds" << std::endl;

    return 0;
}