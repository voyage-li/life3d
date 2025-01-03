/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号: SA24011176
 * 姓名: 李远航
 * 邮箱: voyage@mail.ustc.edu.cn
 ------------------------------------------------*/

#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]
#define AT_BOUNDARY(x, y, z) AT((x + N) % N, (y + N) % N, (z + N) % N)

using std::cin, std::cout, std::endl;
using std::ifstream, std::ofstream;

// 存活细胞数
int population(int N, char* universe) {
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// 打印世界状态
void print_universe(int N, char* universe) {
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                if (AT(x, y, z))
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, universe) << endl;
}

// 核心计算代码，将世界向前推进T个时刻
__global__ void life3d_kernel(int N, char* universe, char* next) {
    __shared__ char shared_universe[10][10][10];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || z >= N)
        return;

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;
    int lz = threadIdx.z + 1;

    shared_universe[lx][ly][lz] = AT(x, y, z);

    const int dx[3] = {-1, 0, 1};
    const int dy[3] = {-1, 0, 1};
    const int dz[3] = {-1, 0, 1};
    const int sx[3] = {0, lx, (int)blockDim.x + 1};
    const int sy[3] = {0, ly, (int)blockDim.y + 1};
    const int sz[3] = {0, lz, (int)blockDim.z + 1};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                if (i == 1 && j == 1 && k == 1)
                    continue;
                if ((i == 0 && lx != 1) || (i == 2 && lx != blockDim.x) || (j == 0 && ly != 1) || (j == 2 && ly != blockDim.y) || (k == 0 && lz != 1) || (k == 2 && lz != blockDim.z))
                    continue;
                if ((i == 0 && lx == 1) || (i == 2 && lx == blockDim.x) ||
                    (j == 0 && ly == 1) || (j == 2 && ly == blockDim.y) ||
                    (k == 0 && lz == 1) || (k == 2 && lz == blockDim.z)) {
                    shared_universe[sx[i]][sy[j]][sz[k]] = AT_BOUNDARY(x + dx[i], y + dy[j], z + dz[k]);
                }
            }
        }
    }

    __syncthreads();

    int alive = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                alive += shared_universe[lx + dx][ly + dy][lz + dz];
            }
        }
    }
    int idx = x * N * N + y * N + z;
    if (shared_universe[lx][ly][lz] && (alive < 5 || alive > 7))
        next[idx] = 0;
    else if (!shared_universe[lx][ly][lz] && alive == 6)
        next[idx] = 1;
    else
        next[idx] = shared_universe[lx][ly][lz];
}

void life3d_run(int N, char* universe, int T) {
    char *d_universe, *d_next;
    cudaMalloc(&d_universe, N * N * N);
    cudaMalloc(&d_next, N * N * N);
    cudaMemcpy(d_universe, universe, N * N * N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.z - 1) / threadsPerBlock.z);
    for (int t = 0; t < T; t++) {
        life3d_kernel<<<numBlocks, threadsPerBlock>>>(N, d_universe, d_next);
        cudaDeviceSynchronize();

        char* temp = d_universe;
        d_universe = d_next;
        d_next = temp;
    }

    cudaMemcpy(universe, d_universe, N * N * N, cudaMemcpyDeviceToHost);

    cudaFree(d_universe);
    cudaFree(d_next);
}

// 读取输入文件
void read_file(char* input_file, char* buffer) {
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size)) {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
void write_file(char* output_file, char* buffer, int N) {
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file) {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char** argv) {
    // cmd args
    if (argc < 5) {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char* input_file = argv[3];
    char* output_file = argv[4];

    char* universe = (char*)malloc(N * N * N);
    read_file(input_file, universe);

    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    life3d_run(N, universe, T);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    write_file(output_file, universe, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    free(universe);
    return 0;
}
