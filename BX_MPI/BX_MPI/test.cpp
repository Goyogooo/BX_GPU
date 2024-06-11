#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <set>
#include <algorithm>
#include <mpi.h>
#include <numeric>
#include <chrono>

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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<uint32_t> small_array, large_array;
    if (world_rank == 0) {
        /*std::ifstream file("D:/MyVS/BX_LW/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "Unable to open file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        file.seekg(32832, std::ios::beg);
        auto array1 = read_array(file);
        auto array2 = read_array(file);
        file.close();
        */
        std::ifstream file("D:/MyVS/BX_LW/ExpIndex", std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件" << std::endl;
            return 1;
        }
        file.seekg(32832, std::ios::beg);
        auto array1 = read_array(file);
        file.seekg(1733008, std::ios::beg);
        auto array2 = read_array(file);
        file.close();

        // Determine which array is smaller
        if (array1.size() <= array2.size()) {
            small_array = array1;
            large_array = array2;
        }
        else {
            small_array = array2;
            large_array = array1;
        }
    }

    // Record the start time
    auto start_time = std::chrono::steady_clock::now();

    // Broadcast the size of the small array
    int small_size = small_array.size();
    MPI_Bcast(&small_size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        small_array.resize(small_size);
    }
    // Broadcast the small array
    MPI_Bcast(small_array.data(), small_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    int large_size;
    if (world_rank == 0) {
        large_size = large_array.size();
    }
    MPI_Bcast(&large_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        large_array.resize(large_size);
    }

    int* sendcounts = new int[world_size];
    int* displs = new int[world_size];
    int remainder = large_array.size() % (world_size-1);
    for (int i = 0; i < world_size-1; ++i) {
        sendcounts[i] = large_array.size() / (world_size-1);  // 每个进程先分配到基本的元素数量
    }
    sendcounts[world_size - 1] = remainder;  // 将所有余数元素添加到最后一个进程的分配中
    int sum = 0;
    for (int i = 0; i < world_size; ++i) {
        displs[i] = sum;          // 计算偏移量
        sum += sendcounts[i];     // 更新sum为下一个偏移量的起点
    }

    std::vector<uint32_t> local_large_array(sendcounts[world_rank]);
    MPI_Scatterv(large_array.data(), sendcounts, displs, MPI_UINT32_T,
        local_large_array.data(), sendcounts[world_rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);

    /*
    // After MPI_Scatterv
    std::cout << "Process " << world_rank << ": local large array size = " << local_large_array.size();
    if (!local_large_array.empty()) {
        std::cout << ", first element = " << local_large_array[0];
    }
    std::cout << std::endl;
    */
    
    std::vector<uint32_t> local_result;
    std::set_intersection(small_array.begin(), small_array.end(),
        local_large_array.begin(), local_large_array.end(),
        std::back_inserter(local_result));
   /*
    // 输出本地交集的大小，用于调试
    int local_size1 = local_result.size();
    std::cout << "Process " << world_rank << " has " << local_size1 << " intersection elements." << std::endl;
    */
    int local_size = local_result.size();
    std::vector<int> recv_sizes(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs_gather(world_size);
    std::vector<uint32_t> global_result;
    if (world_rank == 0) {
        global_result.resize(std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0));
        int offset = 0;
        for (int i = 0; i < world_size; ++i) {
            displs_gather[i] = offset;
            offset += recv_sizes[i];
        }
    }

    MPI_Gatherv(local_result.data(), local_size, MPI_UINT32_T,
        global_result.data(), recv_sizes.data(), displs_gather.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Record the end time
    auto end_time = std::chrono::steady_clock::now();

    // Calculate the elapsed time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();


    if (world_rank == 0) {
        std::cout << "Total results size: " << global_result.size() <<",  time: "<< elapsed_time<< ",  world_size: "<<world_size<<std::endl;
    }

    delete[] sendcounts;
    delete[] displs;
    MPI_Finalize();

    


    

    /*
    // Compute the intersection
    std::vector<uint32_t> local_result;
    std::set_intersection(small_array.begin(), small_array.end(),
        local_large_array.begin(), local_large_array.end(),
        std::back_inserter(local_result));

    // Gather results at the root
    std::vector<uint32_t> global_result;
    if (world_rank == 0) {
        global_result = local_result;
        for (int i = 1; i < world_size; i++) {
            uint32_t size;
            MPI_Recv(&size, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<uint32_t> temp_result(size);
            MPI_Recv(temp_result.data(), size, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_result.insert(global_result.end(), temp_result.begin(), temp_result.end());
        }
    }
    else {
        uint32_t size = local_result.size();
        MPI_Send(&size, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_result.data(), size, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        std::cout << "Total results size: " << global_result.size() << std::endl;
    }

    MPI_Finalize();*/
    return 0;
}
