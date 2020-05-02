#include <algorithm> 
#include <iostream>

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

float* getRowDataDev(float* devData, int row, int length) {
    // float* ret;
    // cudaError_t error = cudaMalloc(&ret, length*sizeof(float));
    // cudaMemcpy(ret, data+row*length, length*sizeof(float), cudaMemcpyHostToDevice);
    return devData + row * length;
}

// Setup

struct GlobalConstants {
    int vocabSize;
    int beamWidth;
    int blankID;
    // int decodeMaxLen;
    char* vocab;
};

__constant__ GlobalConstants cuConstParams;
__device__ int cuNumPaths;

void CTCBeamSearch::setup() {
    // move constants to GPU
    GlobalConstants params;
    params.vocabSize = vocabSize;
    params.beamWidth = beamWidth;
    params.blankID = blankID;
    // params.decodeMaxLen = DECODE_MAX_LEN;
    cudaMalloc(&(params.vocab), vocabSize * sizeof(char));
    cudaMemcpy(params.vocab, this->vocab, vocabSize * sizeof(char), cudaMemcpyHostToDevice);
   
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

    // allocate buffers
    cudaError_t error;
    error = cudaMalloc(&beamStates, beamWidth * vocabSize * sizeof(BeamState*));
    error = cudaMalloc(&nextBeamStates, beamWidth * vocabSize * sizeof(BeamState*));
    error = cudaMalloc(&beamStateBuffer, beamWidth * vocabSize * sizeof(BeamState));
    error = cudaMalloc(&nextBeamStateBuffer, beamWidth * vocabSize * sizeof(BeamState));
    error = cudaMalloc(&pathHashes, beamWidth * vocabSize * sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
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

string CTCBeamSearch::decode(cuMatrix<float>* seqProb){
    setup();
    // get time step
    int timestep = seqProb->getRows();
    // check vocab size
    if(seqProb->getCols() != vocabSize){
        printf("Error: inconsistent vocabulary size in CTC decoder");
        exit(0);
    }

    // initial path at time t = 1
    float* initRow = seqProb->getDev();

    initialPath(getRowDataDev(seqProb->getDev(), 0, vocabSize));
    // initialPath(getRowData(seqProb->getDev(), 0, vocabSize));

    // iterate through timestep
    for (int t = 1; t < timestep; t++){
        float* prob = getRowDataDev(seqProb->getDev(), t, vocabSize);
        extend(prob);
        // path = updatePath;
        // pathScore = updatePathScore;
        // prune();
    }

    // mergeIdenticalPaths();

    // // get best path
    // using pair_type=std::pair<string,float>; 
    // auto bestPath = std::max_element
    // (
    //     std::begin(finalPathScore), std::end(finalPathScore),
    //     [] (const pair_type & p1, const pair_type & p2) {
    //         return p1.second < p2.second;
    //     }
    // );

    // return bestPath->first;
    return "dummy";
}

void CTCBeamSearch::initialPath(float* prob){
    int s = 1;
    BeamState* initialStates[vocabSize];
    cudaError_t error;
    for (int i = 0; i < vocabSize; i++){
        error = cudaMemset(&(beamStateBuffer[i].path), this->vocab[i], sizeof(char));
        error = cudaMemcpy(&(beamStateBuffer[i].prob), &(prob[i]), sizeof(float), cudaMemcpyHostToDevice);
        error = cudaMemcpy(&(beamStateBuffer[i].len), &s, sizeof(int), cudaMemcpyHostToDevice);
        initialStates[i] = beamStateBuffer + i;
    }
    error = cudaMemcpy(beamStates, &initialStates, vocabSize * sizeof(BeamState*), cudaMemcpyHostToDevice);
    
    numPaths = vocabSize;
    error = cudaMemcpyToSymbol(cuNumPaths, &numPaths, sizeof(int));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
    // prune();
}

void CTCBeamSearch::prune(){
    // gather all relevant scores
    vector<float> scores;
    set<string>::iterator iter;
    for (iter=path.begin(); iter!=path.end(); iter++){
        scores.push_back(pathScore.at(*iter));
    }
    
    // sort and get cutoff
    sort (scores.begin(), scores.end(), greater <>());
    float cutoff = scores.at(beamWidth);
    
    // prune
    for (iter=path.begin(); iter!=path.end(); iter++){
        if (pathScore.at(*iter) < cutoff){
            pathScore.erase(*iter);
            iter = path.erase(iter);
        }else{
            ++iter;
        }
    }
}

// TODO: handle length exceeding max len
__global__ void kernelGenNextPaths(float* vocabProbs, BeamState** beamStates, 
    BeamState** nextBeamStates, BeamState* beamStateBuffer, BeamState* nextBeamStateBuffer, int* pathHashes) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int vocabSize = cuConstParams.vocabSize;
    int beamWidth = cuConstParams.beamWidth;
    int blankID = cuConstParams.blankID;
    char* vocab = cuConstParams.vocab;

    if (pid >= vocabSize * cuNumPaths) return;
    
    int pi = pid / vocabSize;
    int vi = pid % vocabSize;

    // char* newPath = nextPathBuffer + decodeMaxLen * pid;
    // nextPaths[pid] = newPath;
    // cudaMemcpy(newPath, paths[pi], decodeMaxLen * sizeof(char), cudaMemcpyDeviceToDevice);
    // nextProbs[pid] = probs[pid] * vocabProbs[vi];

    BeamState* newBeamState = &(nextBeamStateBuffer[pid]);
    BeamState* oldBeamState = beamStates[pi];
    nextBeamStates[pid] = newBeamState;
    char* newPath = newBeamState->path; // Why not the address???
    // printf("%p %p\n", newBeamState, newBeamState->path);    
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
    genHashCode(newPath, newBeamState->len, &(pathHashes[pid]));
}

void CTCBeamSearch::extend(float* vocabProbs){
    // updatePathScore.clear();
    // updatePath.clear();
    // Assume: cudaMalloc initialize memory to zero
    cudaError_t error;
    error = cudaMemset(nextBeamStateBuffer, 0, vocabSize * beamWidth * sizeof(BeamState));
    error = cudaMemset(nextBeamStates, 0, vocabSize * beamWidth * sizeof(BeamState*));
    if (error != cudaSuccess) {
        fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    }
    // kernel
    int blockDim = 256;
    int numBlocks = (numPaths * vocabSize + blockDim - 1) / blockDim;
    kernelGenNextPaths<<<numBlocks, blockDim>>>(vocabProbs, beamStates, 
        nextBeamStates, beamStateBuffer, nextBeamStateBuffer, pathHashes);
    cudaDeviceSynchronize();

    // set<string>::iterator iter;
    // for (iter=path.begin(); iter!=path.end(); iter++){
    //     float score;
    //     string newPath;
    //     for(int i = 0; i < vocabSize; i++){
    //         // extend with blank
    //         if(i == blankID){
    //             // if previous last char is blank
    //             if (string(1, (*iter).back()).compare(vocab[blankID]) == 0){
    //                 newPath = *iter;
    //                 score = pathScore.at(newPath) * prob[blankID];
    //             }else{
    //                 string oldPath = *iter;
    //                 newPath = *iter + vocab[blankID];
    //                 score = pathScore.at(oldPath) * prob[blankID];
    //             }
    //         // extend with symbol
    //         }else{ 
    //             // if previous last char is blank
    //             if (string(1, (*iter).back()).compare(vocab[blankID]) == 0){
    //                 newPath = *iter;
    //                 newPath.replace(newPath.size()-1, 1, vocab.at(i));
    //                 score = pathScore.at(*iter) * prob[i];
    //             }else{
    //                 string lastChar = string(1, (*iter).back());
    //                 if(vocab.at(i).compare(lastChar)==0){
    //                     newPath = *iter;
    //                 }else{
    //                     newPath = *iter + vocab.at(i);
    //                 }

    //                 score = pathScore.at(*iter) * prob[i];
    //             }
    //         }

            // if(updatePath.find(newPath) != updatePath.end()){
            //     updatePathScore[newPath] += score;
            // }else{
            //     updatePath.insert(newPath);
            //     updatePathScore[newPath] = score;
            // }
    //     }
    // }
}





void CTCBeamSearch::mergeIdenticalPaths(){
    // std::cout << "entering mergeIdenticalPaths......" << std::endl;
    // set<string>::iterator iter;
    // for (iter=path.begin(); iter!=path.end(); iter++){
    //     string p = *iter;
    //     // if the last char is blank
    //     string lastChar = string(1, (*iter).back());
    //     if(lastChar.compare(vocab[blankID]) == 0){
    //         p.erase(p.size()-1, 1);
    //     }

    //     if(finalPath.find(p) != finalPath.end()){
    //         finalPathScore[p] += pathScore[p];
    //     }else{
    //         finalPath.insert(p);
    //         finalPathScore[p] = pathScore[p];
    //     }
    // }
}