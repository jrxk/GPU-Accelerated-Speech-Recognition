#include <algorithm> 
#include <iostream>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include "CTCBeamSearch.h"

using namespace std;

// Utility

__device__ int strComp (const char * s1, const char * s2) {
    for(; *s1 == *s2; ++s1, ++s2)
        if(*s1 == 0)
            return 0;
    return *(unsigned char *)s1 > *(unsigned char *)s2 ? -1 : 1;
}


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

__device__ int beamStateComp(const void* p1, const void* p2){
    const struct BeamState *b1 = *(struct BeamState * const *)p1;
    const struct BeamState *b2 = *(struct BeamState * const *)p2;
    if(b1 == NULL && b2 == NULL){
        return 0;
    }else if(b1 == NULL){
        return -1;
    }else if(b2 == NULL){
        return 1;
    }else{
        return strComp(b1->path, b2->path);
    }
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
__global__ void kernelkernel(float* prob, BeamState** state, int* num) {
    // for cuda-gdb
}

__global__ void kernelGenerateSegmentAndIndex(int* segment, int* index, int size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    int value = tid / stride;
    segment[tid] = value;
    index[tid] = tid;
}

__global__ void kernelGenerateSegment(int* segment, int size, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    int value = tid / stride;
    segment[tid] = value;
}

__global__ void kernelUpdateNumPathsPrune(int* batchNumPaths){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int batchSize = cuConstParams.batchSize;
    // int vocabSize = cuConstParams.vocabSize;
    int beamWidth = cuConstParams.beamWidth;
    if (id >= batchSize) return;
    
    // batchNumPaths[id] = beamWidth > vocabSize ? vocabSize : beamWidth;
    batchNumPaths[id] = beamWidth > batchNumPaths[id] ? batchNumPaths[id] : beamWidth;
}

__global__ void kernelUpdateNumPathsExtend(int* batchNumPaths){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int batchSize = cuConstParams.batchSize;
    int vocabSize = cuConstParams.vocabSize;
    if (id >= batchSize) return;
    
    batchNumPaths[id] = batchNumPaths[id] * vocabSize;
}

__global__ void kernelUpdateNumPathsMerge(int* batchNumPaths, int* differentPathTest){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int batchSize = cuConstParams.batchSize;
    int vocabSize = cuConstParams.vocabSize;
    int beamWidth = cuConstParams.beamWidth;
    if (id >= batchSize) return;
    // first batch
    if(id == 0){
        batchNumPaths[id] = differentPathTest[(id + 1) * vocabSize * beamWidth - 1];
    }else{
        batchNumPaths[id] = differentPathTest[(id + 1) * vocabSize * beamWidth - 1] - differentPathTest[id * vocabSize * beamWidth - 1];
    }
}

bool __device__ operator<(device_string lhs, device_string rhs ) {
    char* l = lhs.raw;
    char* r = rhs.raw;

    for( ; *l && *r && *l==*r; )
    {
        ++l;
        ++r;
    }
    return *l < *r;
}

template <class T>
void CTCBeamSearch::batchSortbyStr(int batchSize, thrust::device_vector<device_string> dVecStr, T* &key, T* &keyScratch, BeamState** &originalBeamState, BeamState** &bufferBeamState){
    int blockDim = 256;
    int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelGenerateSegmentAndIndex<<<numBlocks, blockDim>>>(sortSegment, sortIdx, batchSize * beamWidth * vocabSize, beamWidth * vocabSize);
    int totalSize = batchSize * beamWidth * vocabSize;

    // keep an original copy of prob
    // cudaMemcpy(keyScratch, key, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    // sortIdx is mixed decreasing order
    thrust::stable_sort_by_key(thrust::device, dVecStr.begin(),  dVecStr.end(), sortIdx);
    // gather segment to get the corresponding segment for each index
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, sortSegment, sortSegmentScratch);
    std::swap(sortSegment, sortSegmentScratch);
    // sortIdx is the final order (increasing segment, decreasing prob)
    thrust::stable_sort_by_key(thrust::device, sortSegment, sortSegment + totalSize, sortIdx, thrust::less<int>());
    // gather prob by the final order
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, key, keyScratch);
    std::swap(key, keyScratch);
    // gather beam states by the final order
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, originalBeamState, bufferBeamState);
    std::swap(originalBeamState, bufferBeamState);
    kernelkernel<<<1,1>>>(NULL, originalBeamState, NULL);
}

template <class T>
void CTCBeamSearch::batchSortbyKey(int batchSize, T* &key, T* &keyScratch, BeamState** &originalBeamState, BeamState** &bufferBeamState){
    int blockDim = 256;
    int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelGenerateSegmentAndIndex<<<numBlocks, blockDim>>>(sortSegment, sortIdx, batchSize * beamWidth * vocabSize, beamWidth * vocabSize);
    int totalSize = batchSize * beamWidth * vocabSize;

    // keep an original copy of prob
    cudaMemcpy(keyScratch, key, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    // sortIdx is mixed decreasing order
    thrust::stable_sort_by_key(thrust::device, keyScratch, keyScratch + totalSize, sortIdx, thrust::greater<T>());
    // gather segment to get the corresponding segment for each index
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, sortSegment, sortSegmentScratch);
    std::swap(sortSegment, sortSegmentScratch);
    // sortIdx is the final order (increasing segment, decreasing prob)
    thrust::stable_sort_by_key(thrust::device, sortSegment, sortSegment + totalSize, sortIdx, thrust::less<int>());
    // gather prob by the final order
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, key, keyScratch);
    std::swap(key, keyScratch);
    // gather beam states by the final order
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, originalBeamState, bufferBeamState);
    std::swap(originalBeamState, bufferBeamState);
}

void CTCBeamSearch::batchSortByProb(int batchSize) {
    int blockDim = 256;
    int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelGenerateSegmentAndIndex<<<numBlocks, blockDim>>>(sortSegment, sortIdx, batchSize * beamWidth * vocabSize, beamWidth * vocabSize);
    int totalSize = batchSize * beamWidth * vocabSize;
    // keep an original copy of prob
    cudaMemcpy(mergedProbsScratch, mergedProbs, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    // sortIdx is mixed decreasing order
    thrust::stable_sort_by_key(thrust::device, mergedProbsScratch, mergedProbsScratch + totalSize, sortIdx, thrust::greater<float>());
    // gather segment to get the corresponding segment for each index
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, sortSegment, sortSegmentScratch);
    std::swap(sortSegment, sortSegmentScratch);
    // sortIdx is the final order (increasing segment, decreasing prob)
    thrust::stable_sort_by_key(thrust::device, sortSegment, sortSegment + totalSize, sortIdx, thrust::less<int>());
    // gather prob by the final order
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, mergedProbs, mergedProbsScratch);
    std::swap(mergedProbs, mergedProbsScratch);
    // gather beam states by the final order
    thrust::gather(thrust::device, sortIdx, sortIdx + totalSize, beamStates, nextBeamStates);
    std::swap(beamStates, nextBeamStates);
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
    cudaError_t error;
    error = cudaMalloc(&sortIdx, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&sortSegment, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&sortIdxScratch, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&sortSegmentScratch, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&mergedProbsScratch, batchSize * beamWidth * vocabSize * sizeof(float));
    // int blockDim = 256;
    // int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    // kernelGenerateSegmentAndIndex<<<numBlocks, blockDim>>>(sortSegment, sortIdx, batchSize * beamWidth * vocabSize, beamWidth * vocabSize);

    error = cudaMalloc(&beamStates, batchSize * beamWidth * vocabSize * sizeof(BeamState*));
    error = cudaMalloc(&nextBeamStates, batchSize * beamWidth * vocabSize * sizeof(BeamState*));
    error = cudaMalloc(&beamStateBuffer, batchSize * beamWidth * vocabSize * sizeof(BeamState));
    error = cudaMalloc(&nextBeamStateBuffer, batchSize * beamWidth * vocabSize * sizeof(BeamState));
    error = cudaMalloc(&pathHashes, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&pathHashesScratch, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&differentPathTest, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&differentPathTestBuffer, batchSize * beamWidth * vocabSize * sizeof(int));
    error = cudaMalloc(&mergedProbs, batchSize * beamWidth * vocabSize * sizeof(float));
    error = cudaMalloc(&batchNumPaths, batchSize * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
    // dVecPaths.reserve(batchSize * beamWidth * vocabSize);
}

// Decode

vector<pair<string, float>> CTCBeamSearch::decode(cuMatrix<float>* seqProb, int timestep, int batchSize) {
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
        extendAndPrune(prob, t == timestep - 1, batchSize);
    }

    int bestLen;
    char best[DECODE_MAX_LEN];
    BeamState* bestState;
    string best_string;
    float bestScore;
    pair <string,double> resultPair;
    vector<pair<string, float>> bestResults;
    for (int i = 0; i < batchSize; i++){
        cudaMemcpy(&bestState, beamStates + i * beamWidth * vocabSize, sizeof(BeamState*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestLen, &(bestState->len), sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(best, bestState->path, bestLen, cudaMemcpyDeviceToHost);
        best_string = string(best, bestLen);
        cudaMemcpy(&bestScore, &(bestState->prob), sizeof(float), cudaMemcpyDeviceToHost);
        resultPair = make_pair(best_string, bestScore);
        bestResults.push_back(resultPair);
    }
    
    return bestResults;
    // int bestLen;
    // char best[DECODE_MAX_LEN];
    // BeamState* bestState;
    // cudaMemcpy(&bestState, beamStates, sizeof(BeamState*), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&bestLen, &(bestState->len), sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(best, bestState->path, bestLen, cudaMemcpyDeviceToHost);
    // string best_string = string(best, bestLen);
    // float bestScore;
    // cudaMemcpy(&bestScore, &(bestState->prob), sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Best Score: " << bestScore << std::endl; 
    // return best_string;
}

__global__ void decomposeInclusiveScan(int batchSize, int* differentPathTest, int* differentPathTestBuffer, int* batchNumPaths){
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int vocabSize = cuConstParams.vocabSize;
    int beamWidth = cuConstParams.beamWidth;

    if (pid >= batchSize * vocabSize * beamWidth) return;

    int bi = pid / (vocabSize * beamWidth);
    int pi = pid % (vocabSize * beamWidth);
    if (pi >= batchNumPaths[bi]) return; 
    // first batch need not change
    if(bi == 0) {
        differentPathTestBuffer[pid] = differentPathTest[pid] - 1;
        return;
    }

    int offset = bi * (vocabSize * beamWidth);
    differentPathTestBuffer[pid] = differentPathTest[pid] - differentPathTest[bi * vocabSize * beamWidth - 1] + offset -1; 
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
    
    // prune initial path
    // batchSortByProb(batchSize);
    batchSortbyKey<float>(batchSize, mergedProbs, mergedProbsScratch, beamStates, nextBeamStates);
    numBlocks = (batchSize + blockDim - 1) / blockDim;
    kernelUpdateNumPathsPrune<<<numBlocks, blockDim>>>(batchNumPaths);

    // numPaths = vocabSize;
    // thrust::sort_by_key(thrust::device, mergedProbs, mergedProbs + numPaths, beamStates, thrust::greater<float>());
    // numPaths = beamWidth > vocabSize ? vocabSize : beamWidth;

    // if (error != cudaSuccess) {
    //     fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    // }
}

// TODO: handle length exceeding max len
__global__ void kernelGenNextPaths(float* vocabProbs, BeamState** beamStates, 
    BeamState** nextBeamStates, BeamState* beamStateBuffer, BeamState* nextBeamStateBuffer, 
    int* pathHashes, int* batchNumPaths, bool isLastStep, int batchSize) {

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int vocabSize = cuConstParams.vocabSize;
    int beamWidth = cuConstParams.beamWidth;
    int blankID = cuConstParams.blankID;
    char* vocab = cuConstParams.vocab;

    if (pid >= batchSize * vocabSize * beamWidth) return;
    
    int bi = pid / (vocabSize * beamWidth);
    int batchOffset = pid % (vocabSize * beamWidth);
    int pi = batchOffset / vocabSize;
    if (pi >= batchNumPaths[bi]) return;
    int vi = batchOffset % vocabSize;

    BeamState** batchOldBeamStates = beamStates + bi * beamWidth * vocabSize;
    BeamState* newBeamState = &(nextBeamStateBuffer[pid]);
    BeamState* oldBeamState = batchOldBeamStates[pi];
    nextBeamStates[pid] = newBeamState;
    char* newPath = newBeamState->path;
    char* oldPath = oldBeamState->path;
    memcpy(newPath, oldPath, DECODE_MAX_LEN * sizeof(char));
    newBeamState->prob = oldBeamState->prob * vocabProbs[bi * vocabSize + vi];
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

__global__ void kernelTestDifferentPaths(int* pathHashes, int* differentPathTest, int* batchNumPaths, int batchSize) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (pid >= numPaths) return;
    // if (pid == 0 || pathHashes[pid] != pathHashes[pid-1]) differentPathTest[pid] = 1; 
    int beamWidth = cuConstParams.beamWidth;
    int vocabSize = cuConstParams.vocabSize;
    if (pid >= batchSize * beamWidth * vocabSize) return;
    
    int bi = pid / (vocabSize * beamWidth);
    int pi = pid % (vocabSize * beamWidth);
    if (pi >= batchNumPaths[bi]) return;

    if(pi == 0 || pathHashes[pid] != pathHashes[pid - 1]) differentPathTest[pid] = 1;
}



__global__ void kernelMergeSamePaths(int* differentPathTest, BeamState** dest, BeamState** src, float* mergedProbs, int* batchNumPaths, int batchSize) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int beamWidth = cuConstParams.beamWidth;
    int vocabSize = cuConstParams.vocabSize;
    if (pid >= batchSize * beamWidth * vocabSize) return;
    int bi = pid / (vocabSize * beamWidth);
    int pi = pid % (vocabSize * beamWidth);
    if (pi >= batchNumPaths[bi]) return;

    int dstIdx = differentPathTest[pid];
    dest[dstIdx] = src[pid]; 
    atomicAdd(mergedProbs + dstIdx, src[pid]->prob);
}

__global__ void kernelWriteMergedProbs(float* mergedProbs, BeamState** beamStates, int* batchNumPaths) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int beamWidth = cuConstParams.beamWidth;
    int vocabSize = cuConstParams.vocabSize;
    int batchSize = cuConstParams.batchSize;

    if (pid >= batchSize * beamWidth * vocabSize) return;
    int bi = pid / (vocabSize * beamWidth);
    int pi = pid % (vocabSize * beamWidth);
    if (pi >= batchNumPaths[bi]) return;

    beamStates[pid]->prob = mergedProbs[pid];
    // if (pid >= numPaths) return;
    // beamStates[pid]->prob = mergedProbs[pid];
}

bool __device__ devComp(device_string lhs, device_string rhs ) {
    char* l = lhs.raw;
    char* r = rhs.raw;

    for( ; *l && *r && *l==*r; )
    {
        ++l;
        ++r;
    }
    return *l < *r;
}

void CTCBeamSearch::extendAndPrune(float* vocabProbs, bool isLastStep, int batchSize){
    // Assume: cudaMalloc initialize memory to zero
    cudaError_t error;
    error = cudaMemset(mergedProbs, 0, batchSize * vocabSize * beamWidth * sizeof(float));
    // generate all possible new paths (numPaths * vocabSize)
    int blockDim = 256;
    int numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelGenNextPaths<<<numBlocks, blockDim>>>(vocabProbs, beamStates, 
        nextBeamStates, beamStateBuffer, nextBeamStateBuffer, pathHashes, batchNumPaths, isLastStep, batchSize);
    // cudaDeviceSynchronize();

    // sort by hash to group identical paths
    // numPaths = numPaths * vocabSize;
    numBlocks = (batchSize + blockDim - 1) / blockDim;
    kernelUpdateNumPathsExtend<<<numBlocks, blockDim>>>(batchNumPaths);

    thrust::device_vector<device_string> d_vec;
    d_vec.reserve(batchSize * beamWidth * vocabSize);
    for (int i = 0; i < batchSize * beamWidth * vocabSize; i++) {
        device_string d_str(((char*) (nextBeamStateBuffer + i) + 8));
        // auto d_ptr = thrust::device_pointer_cast<BeamState>(beamStateBuffer + i);
        d_vec.push_back(d_str);
    }
    // TODO: sort nextBeamStates directly
    // std::cout << d_vec.size() << std::endl;
    // thrust::sort(thrust::device, d_vec.begin(), d_vec.end());
    // device_string d_str_top = d_vec[0];
    // char d_str_val[255];
    // cudaMemcpy(d_str_val, d_str_top.raw, 255, cudaMemcpyDeviceToHost);
    // std::cout << d_str_val << std::endl;

    
    batchSortbyStr<int>(batchSize, d_vec, pathHashes, pathHashesScratch, nextBeamStates, beamStates);
    // batchSortbyKey<int>(batchSize, pathHashes, pathHashesScratch, nextBeamStates, beamStates);
    
// thrust::sort_by_key(thrust::device, pathHashes, pathHashes + numPaths, nextBeamStates);
    


    // test + scan to get index in merged array (unique paths)
    numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelTestDifferentPaths<<<numBlocks, blockDim>>>(pathHashes, differentPathTest, batchNumPaths, batchSize);
    // thrust::inclusive_scan(thrust::device, differentPathTest, differentPathTest + numPaths, differentPathTest);
    thrust::inclusive_scan(thrust::device, differentPathTest, differentPathTest + batchSize * beamWidth * vocabSize, differentPathTest);
    decomposeInclusiveScan<<<numBlocks, blockDim>>>(batchSize, differentPathTest, differentPathTestBuffer, batchNumPaths);

    // merge the probabilities of identical paths
    error = cudaMemset(beamStates, 0, batchSize * vocabSize * beamWidth * sizeof(BeamState*));
    kernelMergeSamePaths<<<numBlocks, blockDim>>>(differentPathTestBuffer, beamStates, nextBeamStates, mergedProbs, batchNumPaths, batchSize);
    
    // sort by probability
    // error = cudaMemcpy(&numPaths, (void *) (differentPathTest + numPaths - 1), sizeof(int), cudaMemcpyDeviceToHost);
    numBlocks = (batchSize + blockDim - 1) / blockDim;
    kernelUpdateNumPathsMerge<<<numBlocks, blockDim>>>(batchNumPaths, differentPathTest);

    // prune
    batchSortbyKey<float>(batchSize, mergedProbs, mergedProbsScratch, beamStates, nextBeamStates);
    kernelUpdateNumPathsPrune<<<numBlocks, blockDim>>>(batchNumPaths);

    // write merged probablities back to BeamState
    kernelkernel<<<1,1>>>(mergedProbs, beamStates, batchNumPaths);
    numBlocks = (batchSize * beamWidth * vocabSize + blockDim - 1) / blockDim;
    kernelWriteMergedProbs<<<numBlocks, blockDim>>>(mergedProbs, beamStates, batchNumPaths);
    kernelkernel<<<1,1>>>(mergedProbs, beamStates, batchNumPaths);
    // std::swap(beamStates, nextBeamStates);
    std::swap(beamStateBuffer, nextBeamStateBuffer);

    error = cudaMemset(nextBeamStateBuffer, 0, batchSize * vocabSize * beamWidth * sizeof(BeamState));
    error = cudaMemset(nextBeamStates, 0, batchSize * vocabSize * beamWidth * sizeof(BeamState*));
    error = cudaMemset(pathHashes, 0, batchSize * vocabSize * beamWidth * sizeof(int));
    error = cudaMemset(pathHashesScratch, 0, batchSize * vocabSize * beamWidth * sizeof(int));
    error = cudaMemset(differentPathTest, 0, batchSize * vocabSize * beamWidth * sizeof(int));
    error = cudaMemset(differentPathTestBuffer, 0, batchSize * vocabSize * beamWidth * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
}