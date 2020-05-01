#include <vector>
#include <map>
#include <set> 

#include "cuMatrix.h"
// #include "CTCNode.h"

using namespace std;

class CTCBeamSearch 
{
    const vector<string> &vocab; // vocab should include blank
    int beamWidth;
    int blankID;
    int vocabSize;
    map<string, float> pathScore;
    set<string> path;

    // intermiate result
    map<string, float> updatePathScore;
    set<string> updatePath;

    map<string, float> finalPathScore;
    set<string> finalPath;


public:
  CTCBeamSearch(const vector<string> &vocab, int beamWidth, int blankID):vocab(vocab), beamWidth(beamWidth), blankID(blankID){
      vocabSize = vocab.size();
  };

  string decode(cuMatrix<float>* seqProb); // assume prob is [seq,vocab] for now (no batch)
  void helper();

  void initialPath(float* prob); 
  void prune();
  void extend(float* prob);
  void mergeIdenticalPaths();

};
