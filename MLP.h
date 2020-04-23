class MLP {
public:

    MLP() {
        
    }

    ~MLP() {

    }

    void forward();

    void init();

private:
    cuMatrix<float> *inputs;

    cuMatrix<float> *weight; 
    cuMatrix<float> *bias; 

};