#include <algorithm> 
#include <iostream>

#include "CTCBeamSearch.h"

using namespace std;

float* getRowData(float* data, int row, int length){
    float* ret= new float[length];
    memcpy(ret, data+row*length, length*sizeof(float));
    return ret;
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
    // get time step
    int timestep = seqProb->getRows();
    // check vocab size
    if(seqProb->getCols() != vocabSize){
        printf("Error: inconsistent vocabulary size in CTC decoder");
        exit(0);
    }

    // initial path at time t = 1
    initialPath(getRowData(seqProb->getHost(), 0, vocabSize));
    // initialPath(getRowData(seqProb->getDev(), 0, vocabSize));

    // iterate through timestep
    for (int t = 1; t < timestep; t++){
        float* prob = getRowData(seqProb->getHost(), t, vocabSize);
        extend(prob);
        path = updatePath;
        pathScore = updatePathScore;
        prune();
    }

    mergeIdenticalPaths();

    // get best path
    using pair_type=std::pair<string,float>; 
    auto bestPath = std::max_element
    (
        std::begin(finalPathScore), std::end(finalPathScore),
        [] (const pair_type & p1, const pair_type & p2) {
            return p1.second < p2.second;
        }
    );

    return bestPath->first;
}

void CTCBeamSearch::initialPath(float* prob){
    for (int i = 0; i < vocabSize; i++){
        string initpath = vocab.at(i);
        path.insert(initpath);
        pathScore.insert(pair<string, float>(initpath, prob[i]));
    }

    prune();
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

void CTCBeamSearch::extend(float* prob){
    updatePathScore.clear();
    updatePath.clear();

    set<string>::iterator iter;
    for (iter=path.begin(); iter!=path.end(); iter++){
        float score;
        string newPath;
        for(int i = 0; i < vocabSize; i++){
            // extend with blank
            if(i == blankID){
                // if previous last char is blank
                if (string(1, (*iter).back()).compare(vocab[blankID]) == 0){
                    newPath = *iter;
                    score = pathScore.at(newPath) * prob[blankID];
                }else{
                    string oldPath = *iter;
                    newPath = *iter + vocab[blankID];
                    score = pathScore.at(oldPath) * prob[blankID];
                }
            // extend with symbol
            }else{ 
                // if previous last char is blank
                if (string(1, (*iter).back()).compare(vocab[blankID]) == 0){
                    newPath = *iter;
                    newPath.replace(newPath.size()-1, 1, vocab.at(i));
                    score = pathScore.at(*iter) * prob[i];
                }else{
                    string lastChar = string(1, (*iter).back());
                    if(vocab.at(i).compare(lastChar)==0){
                        newPath = *iter;
                    }else{
                        newPath = *iter + vocab.at(i);
                    }

                    score = pathScore.at(*iter) * prob[i];
                }
            }

            if(updatePath.find(newPath) != updatePath.end()){
                updatePathScore[newPath] += score;
            }else{
                updatePath.insert(newPath);
                updatePathScore[newPath] = score;
            }
        }
    }
}

void CTCBeamSearch::mergeIdenticalPaths(){
    // std::cout << "entering mergeIdenticalPaths......" << std::endl;
    set<string>::iterator iter;
    for (iter=path.begin(); iter!=path.end(); iter++){
        string p = *iter;
        // if the last char is blank
        string lastChar = string(1, (*iter).back());
        if(lastChar.compare(vocab[blankID]) == 0){
            p.erase(p.size()-1, 1);
        }

        if(finalPath.find(p) != finalPath.end()){
            finalPathScore[p] += pathScore[p];
        }else{
            finalPath.insert(p);
            finalPathScore[p] = pathScore[p];
        }
    }
}