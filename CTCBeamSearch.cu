#include <algorithm> 
#include <iostream>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "CTCBeamSearch.h"

using namespace std;

// Utility

// Hash code C strings (same as Java hashcode())
int strHashCode(char* str, int len) {
    int hash = 0;
    if (len == 0) return hash;
    for (int i = 0; i < len; i++) {
        char c = str[i];
        hash = (31 * hash) + c;
    }
    return hash;
}

__device__ inline void genHashCode(char* str, int len, int* dest) {
    int hash = 0;
    if (len == 0) *dest = hash;
    for (int i = 0; i < len; i++) {
        char c = str[i];
        hash = (31 * hash) + c;
    }
    *dest = hash;
}

float* getRowData(float* data, int row, int length){
    float* ret= new float[length];
    memcpy(ret, data+row*length, length*sizeof(float));
    return ret;
}

// seq_len, 
float* getBatchAtT(float* devData, int timeIdx, int batchSize, int length) {
    return devData + timeIdx * batchSize * length;
}

void printMap (map<string, float> dict){
    for(map<string, float >::const_iterator it = dict.begin(); it != dict.end(); ++it)
    {
        std::cout << it->first << "," << it->second << ";";
    }

    std::cout<<endl;
}

void printSet (set<string> myset){
    std::set<std::string>::iterator it = myset.begin();
    while (it != myset.end())
    {
        std::cout << (*it) << ",";
        it++;
    }
    std::cout<<endl;
}

void printVector (vector<float> vec){
    std::vector<float>::iterator it = vec.begin();
    while (it != vec.end())
    {
        std::cout << (*it) << ",";
        it++;
    }
    std::cout<<endl;
}

void CTCBeamSearch::helper(){
    std::cout << "======print path======" << std::endl;
    printSet(path);
    std::cout << "======print pathScore======" << std::endl;
    printMap(pathScore);
}

// Setup

struct GlobalConstants {
    int vocabSize;
    int beamWidth;
    int blankID;
    // int decodeMaxLen;
    int batchSize;
    char* vocab;
};

__constant__ GlobalConstants cuConstParams;
// __device__ int cuNumPaths;

__device__ int my_mod_start = 0;
__device__ int my_mod(){
    return (my_mod_start++)/8;
}

__global__ void kernelGenerateSegmentAndIndex(int* segment, int* index, int size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    int value = tid / stride;
    segment[tid] = value;
    index[tid] = tid;
}

void CTCBeamSearch::batchSortByProb(float* batchProb, BeamState** beamStates, int* sortIdx, int* sortSegment, int batchSize) {
    int blockDim = 256;
    int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelGenerateSegmentAndIndex<<<numBlocks, blockDim>>>(sortSegment, sortIdx, batchSize * beamWidth * vocabSize, beamWidth * vocabSize);
    int totalSize = batchSize * beamWidth * vocabSize;
    thrust::stable_sort_by_key(thrust::device, batchProb, batchProb + totalSize, sortIdx, thrust::greater<float>());
    // thrust::gather (need a different gather destination array)
}

void CTCBeamSearch::setup(int batchSize) {
    // move constants to GPU
    GlobalConstants params;
    params.vocabSize = vocabSize;
    params.beamWidth = beamWidth;
    params.blankID = blankID;
    params.batchSize = batchSize;
    // params.decodeMaxLen = DECODE_MAX_LEN;
    cudaMalloc(&(params.vocab), vocabSize * sizeof(char));
    cudaMemcpy(params.vocab, this->vocab, vocabSize * sizeof(char), cudaMemcpyHostToDevice);
   
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

    // allocate buffers
    cudaMalloc(&sortIdx, batchSize * beamWidth * vocabSize * sizeof(int));
    cudaMalloc(&sortSegment, batchSize * beamWidth * vocabSize * sizeof(int));
    // int blockDim = 256;
    // int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    // kernelGenerateSegmentAndIndex<<<numBlocks, blockDim>>>(sortSegment, sortIdx, batchSize * beamWidth * vocabSize, beamWidth * vocabSize);
    
    cudaError_t error;
    error = cudaMalloc(&beamStates, batchSize * beamWidth * vocabSize * sizeof(BeamState*));
    error = cudaMalloc(&nextBeamStates, batchSize * beamWidth * vocabSize * sizeof(BeamState*));
    error = cudaMalloc(&beamStateBuffer, batchSize * beamWidth * vocabSize * sizeof(BeamState));
    error = cudaMalloc(&nextBeamStateBuffer, batchSize * beamWidth * vocabSize * sizeof(BeamState));
    error = cudaMalloc(&pathHashes, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&differentPathTest, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&mergedProbs, batchSize * beamWidth * vocabSize * sizeof(float));
    error = cudaMalloc(&batchNumPaths, batchSize * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
}

// Decode

string CTCBeamSearch::decode(cuMatrix<float>* seqProb, int timestep, int batchSize) {
    setup(batchSize);
    // get time step
    // int timestep = seqProb->getRows();
    // check vocab size
    if(seqProb->getCols() != vocabSize){
        printf("Error: inconsistent vocabulary size in CTC decoder");
        exit(0);
    }

    // initial path at time t = 1
    float* initRow = seqProb->getDev();

    initialPath(getBatchAtT(seqProb->getDev(), 0, batchSize, vocabSize), batchSize);

    // iterate through timestep
    for (int t = 1; t < timestep; t++){
        float* prob = getBatchAtT(seqProb->getDev(), t, batchSize, vocabSize);
        extendAndPrune(prob, t == timestep - 1);
    }

    int bestLen;
    char best[DECODE_MAX_LEN];
    BeamState* bestState;
    cudaMemcpy(&bestState, beamStates, sizeof(BeamState*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bestLen, &(bestState->len), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(best, bestState->path, bestLen, cudaMemcpyDeviceToHost);
    string best_string = string(best, bestLen);
    float bestScore;
    cudaMemcpy(&bestScore, &(bestState->prob), sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Best Score: " << bestScore << std::endl; 
    return best_string;
}

// prob: batchSize, vocabSize
// beamStates, ...: batchSize, beamWidth * vocabSize
// batchNumPaths: batchSize,
__global__ void kernelInitialPath(float* prob, BeamState** beamStates, BeamState* beamStateBuffer, float* mergedProbs,
    int* batchNumPaths) {
    
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int batchSize = cuConstParams.batchSize;
    int vocabSize = cuConstParams.vocabSize;
    if (pid >= vocabSize * batchSize) return;

    int beamWidth = cuConstParams.beamWidth;
    int blankID = cuConstParams.blankID;
    char* vocab = cuConstParams.vocab;

    int exampleIdx = pid / vocabSize; // example index inside each batch 
    int pathIdx = pid % vocabSize; // initialize vocabSize paths for each example in batch

    BeamState** exampleBeamStates = beamStates + exampleIdx * (beamWidth * vocabSize);
    BeamState* exampleBeamStateBuffer = beamStateBuffer + exampleIdx * (beamWidth * vocabSize);
    float* exampleMergedProbs = mergedProbs + exampleIdx * (beamWidth * vocabSize);

    exampleBeamStateBuffer[pathIdx].path[0] = vocab[pathIdx];
    float currProb = prob[exampleIdx * vocabSize + pathIdx];
    exampleBeamStateBuffer[pathIdx].prob = currProb;
    exampleBeamStateBuffer[pathIdx].len = 1;
    exampleBeamStates[pathIdx] = exampleBeamStateBuffer + pathIdx;
    
    exampleMergedProbs[pathIdx] = currProb;
    batchNumPaths[exampleIdx] = vocabSize;
}

void CTCBeamSearch::initialPath(float* prob, int batchSize) {
    // int s = 1;
    // BeamState* initialStates[vocabSize];
    // cudaError_t error;
    // for (int i = 0; i < vocabSize; i++){
    //     error = cudaMemset(&(beamStateBuffer[i].path), this->vocab[i], sizeof(char));
    //     error = cudaMemcpy(&(beamStateBuffer[i].prob), &(prob[i]), sizeof(float), cudaMemcpyDeviceToDevice);
    //     error = cudaMemcpy(&(beamStateBuffer[i].len), &s, sizeof(int), cudaMemcpyHostToDevice);
    //     initialStates[i] = beamStateBuffer + i;
    // }
    // error = cudaMemcpy(beamStates, &initialStates, vocabSize * sizeof(BeamState*), cudaMemcpyHostToDevice);
    
    // numPaths = vocabSize;
    // // error = cudaMemcpyToSymbol(cuNumPaths, &numPaths, sizeof(int));

    // // prune
    // error = cudaMemcpy(mergedProbs, prob, vocabSize * sizeof(float), cudaMemcpyDeviceToDevice);
    

    int blockDim = 256;
    int numBlocks = (batchSize * vocabSize + blockDim - 1) / blockDim;
    kernelInitialPath<<<numBlocks, blockDim>>>(prob, beamStates, beamStateBuffer, mergedProbs, batchNumPaths);
    
    numPaths = vocabSize;
    thrust::sort_by_key(thrust::device, mergedProbs, mergedProbs + numPaths, beamStates, thrust::greater<float>());
    numPaths = beamWidth > vocabSize ? vocabSize : beamWidth;

    // if (error != cudaSuccess) {
    //     fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    // }
}

// TODO: handle length exceeding max len
__global__ void kernelGenNextPaths(float* vocabProbs, BeamState** beamStates, 
    BeamState** nextBeamStates, BeamState* beamStateBuffer, BeamState* nextBeamStateBuffer, 
    int* pathHashes, int numPaths, bool isLastStep) {

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int vocabSize = cuConstParams.vocabSize;
    // int beamWidth = cuConstParams.beamWidth;
    int blankID = cuConstParams.blankID;
    char* vocab = cuConstParams.vocab;

    if (pid >= vocabSize * numPaths) return;
    
    int pi = pid / vocabSize;
    int vi = pid % vocabSize;

    BeamState* newBeamState = &(nextBeamStateBuffer[pid]);
    BeamState* oldBeamState = beamStates[pi];
    nextBeamStates[pid] = newBeamState;
    char* newPath = newBeamState->path;
    char* oldPath = oldBeamState->path;
    memcpy(newPath, oldPath, DECODE_MAX_LEN * sizeof(char));
    newBeamState->prob = oldBeamState->prob * vocabProbs[vi];
    // extend with blank
    if (vi == blankID) {
        // path last char is blank
        if (oldPath[oldBeamState->len-1] == vocab[blankID]) {
            newBeamState->len = oldBeamState->len;
        } else {
            newBeamState->len = oldBeamState->len + 1;
            newPath[newBeamState->len-1] = vocab[vi]; // append new blank
        }
    } else {
        if (oldPath[oldBeamState->len-1] == vocab[blankID]) {
            newBeamState->len = oldBeamState->len;
            newPath[newBeamState->len-1] = vocab[vi]; // replace last blank with new char
        } else {
            if (oldPath[oldBeamState->len-1] == vocab[vi]) {
                newBeamState->len = oldBeamState->len;
            } else {
                newBeamState->len = oldBeamState->len + 1;
                newPath[newBeamState->len-1] = vocab[vi]; // append new char
            }
        }
    }
    if (isLastStep) {
        if (newPath[newBeamState->len-1] == vocab[blankID]) {
            newBeamState->len -= 1;
        }
    }
    genHashCode(newPath, newBeamState->len, &(pathHashes[pid]));
}

__global__ void kernelTestDifferentPaths(int* pathHashes, int* differentPathTest, int numPaths) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numPaths) return;
    if (pid == 0 || pathHashes[pid] != pathHashes[pid-1]) differentPathTest[pid] = 1; 
}

__global__ void kernelkernel(float* a1, BeamState** a2) {
    // for cuda-gdb
}

__global__ void kernelMergeSamePaths(int* differentPathTest, BeamState** dest, BeamState** src, float* mergedProbs, int numPaths) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numPaths) return;
    int dstIdx = differentPathTest[pid] - 1;
    dest[dstIdx] = src[pid]; 
    atomicAdd(mergedProbs + dstIdx, src[pid]->prob);
}

__global__ void kernelWriteMergedProbs(float* mergedProbs, BeamState** beamStates, int numPaths) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numPaths) return;
    beamStates[pid]->prob = mergedProbs[pid];
}

void CTCBeamSearch::extendAndPrune(float* vocabProbs, bool isLastStep){
    // Assume: cudaMalloc initialize memory to zero
    cudaError_t error;
    error = cudaMemset(mergedProbs, 0, vocabSize * beamWidth * sizeof(float));
    // generate all possible new paths (numPaths * vocabSize)
    int blockDim = 256;
    int numBlocks = (numPaths * vocabSize + blockDim - 1) / blockDim;
    kernelGenNextPaths<<<numBlocks, blockDim>>>(vocabProbs, beamStates, 
        nextBeamStates, beamStateBuffer, nextBeamStateBuffer, pathHashes, numPaths, isLastStep);
    
    // cudaDeviceSynchronize();

    // sort by hash to group identical paths
    numPaths = numPaths * vocabSize;
    thrust::sort_by_key(thrust::device, pathHashes, pathHashes + numPaths, nextBeamStates);
    // test + scan to get index in merged array (unique paths)
    numBlocks = (numPaths * vocabSize + blockDim - 1) / blockDim;
    kernelTestDifferentPaths<<<numBlocks, blockDim>>>(pathHashes, differentPathTest, numPaths);
    thrust::inclusive_scan(thrust::device, differentPathTest, differentPathTest + numPaths, differentPathTest);
    // merge the probabilities of identical paths
    error = cudaMemset(beamStates, 0, vocabSize * beamWidth * sizeof(BeamState*));
    kernelMergeSamePaths<<<numBlocks, blockDim>>>(differentPathTest, beamStates, nextBeamStates, mergedProbs, numPaths);
    // sort by probability
    error = cudaMemcpy(&numPaths, (void *) (differentPathTest + numPaths - 1), sizeof(int), cudaMemcpyDeviceToHost);
    thrust::sort_by_key(thrust::device, mergedProbs, mergedProbs + numPaths, beamStates, thrust::greater<float>());
    // prune
    numPaths = beamWidth > numPaths ? numPaths : beamWidth;
    // write merged probablities back to BeamState
    numBlocks = (numPaths * vocabSize + blockDim - 1) / blockDim;
    kernelWriteMergedProbs<<<numBlocks, blockDim>>>(mergedProbs, beamStates, numPaths);

    // std::swap(beamStates, nextBeamStates);
    std::swap(beamStateBuffer, nextBeamStateBuffer);

    error = cudaMemset(nextBeamStateBuffer, 0, vocabSize * beamWidth * sizeof(BeamState));
    error = cudaMemset(nextBeamStates, 0, vocabSize * beamWidth * sizeof(BeamState*));
    error = cudaMemset(pathHashes, 0, vocabSize * beamWidth * sizeof(int));
    error = cudaMemset(differentPathTest, 0, vocabSize * beamWidth * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
}