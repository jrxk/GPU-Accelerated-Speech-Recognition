#include <vector>
#include <map>
#include <set> 

#include "cuMatrix.h"
#include "CTCNode.h"

using namespace std;

class CTCBeamSearch 
{
    const vector<string> &vocab; // vocab should include blank
    // const std::vector<std::vector<double>> &probs_seq;
    int beamWidth;
    int blankID;
    int vocabSize;
    map<string, float> pathScore;
    map<string, float> blankPathScore;
    map<string, float> finalPathScore;
    set<string> path;
    set<string> blankPath;

    // intermiate result
    map<string, float> updateBlankPathScore;
    set<string> updateBlankPath;
    map<string, float> updateSymbolPathScore;
    set<string> updateSymbolPath;

public:
    
  CTCBeamSearch(const vector<string> &vocab, int beamWidth, int blankID):vocab(vocab), beamWidth(beamWidth), blankID(blankID){
      vocabSize = vocab.size();
  };

  string decode(cuMatrix<float>* seqProb); // assume prob is [seq,vocab] for now (no batch)
  void helper();

  void initialPath(float* prob); 
  void prune();
  void extendWithBlank(float* prob);
  void extendWithSymbol(float* prob);
  void mergeIdenticalPaths();

};