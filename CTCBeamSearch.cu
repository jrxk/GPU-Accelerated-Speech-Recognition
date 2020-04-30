#include <algorithm> 
#include "CTCBeamSearch.h"

using namespace std;

bool compare (int i,int j) { return (i>j); }
float* getRowData(float* data, int row, int length){
    float* ret= new float[length];
    memcpy(ret, data+row*length, length*sizeof(float));
    return ret;
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
        // get current timestep data (a row)
        // float* prob = getRowData(seqProb->getDev(), t, vocabSize);
        float* prob = getRowData(seqProb->getHost(), t, vocabSize);
        extendWithBlank(prob);
        extendWithSymbol(prob);
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

    prune();
}

void CTCBeamSearch::prune(){
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
}

void CTCBeamSearch::extendWithBlank(float* prob){
    map<string, float> updateBlankPathScore;
    set<string> updateBlankPath;

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
    blankPathScore = updateBlankPathScore;
    blankPath = updateBlankPath;
}

void CTCBeamSearch::extendWithSymbol(float* prob){
    map<string, float> updateSymbolPathScore;
    set<string> updateSymbolPath;

    set<string>::iterator iterBlank;
    for (iterBlank=blankPath.begin(); iterBlank!=blankPath.end(); iterBlank++){
        for (int i = 0; i < vocabSize; i++){
            // skip if blank
            if(i == blankID){
                continue;
            }

            string newPath = *iterBlank + vocab.at(i);
            float score = blankPathScore.at(newPath) * prob[i];
            updateSymbolPath.insert(newPath);
            updateSymbolPathScore[newPath] = score;
        }
    }

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

            float score = pathScore.at(newPath) * prob[i];
            if(updateSymbolPath.find(newPath) != updateSymbolPath.end()){
                updateSymbolPathScore[newPath] += score;
            }else{
                updateSymbolPath.insert(newPath);
                updateSymbolPathScore[newPath] = score;
            }
        }
    }

    // update pathScore, path
    pathScore = updateSymbolPathScore;
    path = updateSymbolPath;
}

void CTCBeamSearch::mergeIdenticalPaths(){
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