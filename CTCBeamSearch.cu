#include <algorithm> 
#include <iostream>

#include "CTCBeamSearch.h"

using namespace std;

bool compare (int i,int j) { return (i>j); }
float* getRowData(float* data, int row, int length){
    float* ret= new float[length];
    memcpy(ret, data+row*length, length*sizeof(float));
    return ret;
}



void printMap (map<string, float> dict){
    for(map<string, float >::const_iterator it = dict.begin(); it != dict.end(); ++it)
    {
        std::cout << it->first << "," << it->second << "\n";
    }
}

void printSet (set<string> myset){
    std::set<std::string>::iterator it = myset.begin();
    while (it != myset.end())
    {
        std::cout << (*it) << "," << "\n";
        it++;
    }
}

void CTCBeamSearch::helper(){
    std::cout << "======print blankPath======" << std::endl;
    printSet(blankPath);
    std::cout << "======print blankPathScore======" << std::endl;
    printMap(blankPathScore);
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
        std::cout << "time step: " << t << std::endl;
        // get current timestep data (a row)
        // float* prob = getRowData(seqProb->getDev(), t, vocabSize);
        float* prob = getRowData(seqProb->getHost(), t, vocabSize);
        extendWithBlank(prob);
        extendWithSymbol(prob);
        blankPath = updateBlankPath;
        blankPathScore = updateBlankPathScore;
        path = updateSymbolPath;
        pathScore = updateSymbolPathScore;
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
    std::cout << "entering initialPath......" << std::endl;
    string initpath = "";
    blankPathScore.insert(pair<string, float>(initpath, prob[blankID]));
    blankPath.insert(initpath);

    for (int i = 0; i < vocabSize; i++){
        // skip if blank
        if(i == blankID){
            continue;
        }

        initpath = vocab.at(i);
        pathScore.insert(pair<string, float>(initpath, prob[i]));
        path.insert(initpath);
    }

    helper();

    prune();
}

void CTCBeamSearch::prune(){
    std::cout << "entering prune......" << std::endl;

    vector<float> scores;
    // gather all relevant scores
    set<string>::iterator iterBlank;
    for (iterBlank=blankPath.begin(); iterBlank!=blankPath.end(); iterBlank++){
        scores.push_back(blankPathScore.at(*iterBlank));
    }

    set<string>::iterator iterSymbol;
    for (iterSymbol=path.begin(); iterSymbol!=path.end(); iterSymbol++){
        scores.push_back(pathScore.at(*iterSymbol));
    }

    // sort and get cutoff
    sort (scores.begin(), scores.end(), compare);
    float cutoff = scores.at(beamWidth);
    
    // prune
    for (iterBlank=blankPath.begin(); iterBlank!=blankPath.end(); ){
        if(blankPathScore.at(*iterBlank) < cutoff){
            blankPathScore.erase(*iterBlank);
            iterBlank = blankPath.erase(iterBlank);
        }else{
            ++iterBlank;
        }
    }

    for (iterSymbol=path.begin(); iterSymbol!=path.end(); ){
        if(pathScore.at(*iterSymbol) < cutoff){
            pathScore.erase(*iterSymbol);
            iterSymbol = path.erase(iterSymbol);
        }else{
            ++iterSymbol;
        }
    }

    helper();
    
}

void CTCBeamSearch::extendWithBlank(float* prob){
    std::cout << "entering extendWithBlank......" << std::endl;

    // map<string, float> updateBlankPathScore;
    // set<string> updateBlankPath;
    updateBlankPathScore.clear();
    updateBlankPath.clear();

    set<string>::iterator iterBlank;
    for (iterBlank=blankPath.begin(); iterBlank!=blankPath.end(); iterBlank++){
        string newPath = *iterBlank;
        float score = blankPathScore.at(newPath) * prob[blankID];
        updateBlankPath.insert(newPath);
        updateBlankPathScore[newPath] = score;
        
        // if(updateBlankPath.find(newPath) != updateBlankPath.end()){
        //     updateBlankPathScore[newPath] += score;
        // }else{
        //     updateBlankPath.insert(newPath);
        //     updateBlankPathScore[newPath] = score;
        // }
    }

    set<string>::iterator iterSymbol;
    for (iterSymbol=path.begin(); iterSymbol!=path.end(); iterSymbol++){
        string newPath = *iterSymbol;
        float score = pathScore.at(newPath) * prob[blankID];
        if(updateBlankPath.find(newPath) != updateBlankPath.end()){
            updateBlankPathScore[newPath] += score;
        }else{
            updateBlankPath.insert(newPath);
            updateBlankPathScore[newPath] = score;
        }
    }

    // update blankPathScore, blankPath
    // blankPathScore = updateBlankPathScore;
    // blankPath = updateBlankPath;
}

void CTCBeamSearch::extendWithSymbol(float* prob){
    std::cout << "entering extendWithSymbol......" << std::endl;
    std::cout << "print blankPath: " << std::endl;
    printSet(blankPath);
    std::cout << "print blankPathScore: " << std::endl;
    printMap(blankPathScore);
    std::cout << "entering extendWithSymbol blank part......" << std::endl;


    // map<string, float> updateSymbolPathScore;
    // set<string> updateSymbolPath;

    updateSymbolPathScore.clear();
    updateSymbolPath.clear();

    set<string>::iterator iterBlank;
    for (iterBlank=blankPath.begin(); iterBlank!=blankPath.end(); iterBlank++){
        std::cout << "current blank path: " << *iterBlank << std::endl;

        for (int i = 0; i < vocabSize; i++){
            std::cout << "index vocab: " << i << std::endl;
            
            // skip if blank
            if(i == blankID){
                continue;
            }

            string newPath = *iterBlank + vocab.at(i);
            std::cout << "newPath: " << newPath << std::endl;

            float score = blankPathScore.at(*iterBlank) * prob[i];
            updateSymbolPath.insert(newPath);
            updateSymbolPathScore[newPath] = score;
        }
    }

    std::cout << "entering extendWithSymbol symbol part......" << std::endl;

    set<string>::iterator iterSymbol;
    for (iterSymbol=path.begin(); iterSymbol!=path.end(); iterSymbol++){
        for (int i = 0; i < vocabSize; i++){
            // skip if blank
            if(i == blankID){
                continue;
            }

            string newPath;
            string lastChar = string(1, (*iterSymbol).back());
            if(vocab.at(i).compare(lastChar)==0){
                newPath = *iterSymbol;
            }else{
                newPath = *iterSymbol + vocab.at(i);
            }

            float score = pathScore.at(*iterSymbol) * prob[i];
            if(updateSymbolPath.find(newPath) != updateSymbolPath.end()){
                updateSymbolPathScore[newPath] += score;
            }else{
                updateSymbolPath.insert(newPath);
                updateSymbolPathScore[newPath] = score;
            }
        }
    }

    // update pathScore, path
    // pathScore = updateSymbolPathScore;
    // path = updateSymbolPath;
}

void CTCBeamSearch::mergeIdenticalPaths(){
    std::cout << "entering mergeIdenticalPaths......" << std::endl;

    set<string>::iterator iterSymbol;
    for (iterSymbol=path.begin(); iterSymbol!=path.end(); iterSymbol++){
        finalPathScore[*iterSymbol] = pathScore[*iterSymbol];
    }

    set<string>::iterator iterBlank;
    for (iterBlank=blankPath.begin(); iterBlank!=blankPath.end(); iterBlank++){
        string p = *iterBlank;
        if(path.find(p) != path.end()){
            finalPathScore[p] += blankPathScore[p];
        }else{
            path.insert(p);
            finalPathScore[p] = blankPathScore[p];
        }
    }
}