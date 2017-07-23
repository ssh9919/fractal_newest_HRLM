/*
   Copyright 2015 Kyuyeon Hwang (kyuyeon.hwang@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#ifndef FRACTAL_MULTICLASSIFPROBECHAR_H_
#define FRACTAL_MULTICLASSIFPROBECHAR_H_


#include "../core/TrainableProbe.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class MultiClassifProbeChar : public TrainableProbe
{
public:
    MultiClassifProbeChar();

    void SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo);
    void Setlambda(FLOAT Inlambda);
    void ComputeErr(const unsigned long idxFrom, const unsigned long idxTo);
    void EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output);
    void ResetStatistics();
    const double GetLoss();
    void PrintStatistics(std::ostream &outStream);

    const unsigned long GetnOOV();
    const unsigned long GetnSample();
    const double GetMeanSquaredError();
    const double GetWordPerplexity();
    const double GetAverageCrossEntropy();
    const double GetFrameErrorRate();

protected:
    virtual void SetEngine(Engine *engine);
    Engine * engine;
    Matrix<FLOAT> target;
    Matrix<FLOAT> gradientMask;
    Matrix<FLOAT> lambdaMat;
    unsigned long nSample;
    unsigned long nError;
    double seSum;
    double ceSum;
    FLOAT lambda;
};

}

#endif /* FRACTAL_MULTICLASSIFPROBECHAR_H_ */

