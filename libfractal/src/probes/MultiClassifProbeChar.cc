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


#include "MultiClassifProbeChar.h"

#include <cmath>

#include "../core/Layer.h"


namespace fractal
{


MultiClassifProbeChar::MultiClassifProbeChar() : TrainableProbe(true)
{
    ResetStatistics();
}

void MultiClassifProbeChar::Setlambda(FLOAT Inlambda)
{
	this->lambda = Inlambda;
}

void MultiClassifProbeChar::SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo)
{
   //printf("(char) start SetTarget\n");
   //fflush(stdout);
    unsigned long nRows, nCols;

    verify(engine != NULL);
    verify(linkedLayer->GetSize() > 1);
    
    verify(mat.GetDataType() == MultiTypeMatrix::DATATYPE_INT);

    nRows = linkedLayer->GetSize();
    nCols = nStream * nUnroll;

    target.Resize(nRows, nCols);
    gradientMask.Resize(nRows, nCols);
    lambdaMat.Resize(nRows, nCols);


    Matrix<FLOAT> targetSub(target, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> gradientMaskSub(gradientMask, idxFrom * nStream, idxTo * nStream + nStream - 1);

    //printf("targetMat nRows: %d\n", target.GetNumRows());
    //printf("targetMat nCols: %d\n", target.GetNumCols());
    //printf("gradientMas nRows: %d\n", gradientMask.GetNumRows());
    //printf("gradientMas nCols: %d\n", gradientMask.GetNumCols());


    switch(mat.GetDataType())
    {
        case MultiTypeMatrix::DATATYPE_FLOAT:
            {
                Matrix<FLOAT> *ptrMat = reinterpret_cast<Matrix<FLOAT> *>(mat.GetMatrix());

                verify(ptrMat->GetNumRows() == nRows);
                verify(ptrMat->GetNumCols() == nStream * (idxTo - idxFrom + 1));

                engine->MatCopy(*ptrMat, targetSub, stream);
            }
            break;

        case MultiTypeMatrix::DATATYPE_INT:
            {
                Matrix<INT> *ptrMat = reinterpret_cast<Matrix<INT> *>(mat.GetMatrix());

                verify(ptrMat->GetNumRows() == 2); //change 1 to 2 for using mask 
                verify(ptrMat->GetNumCols() == nStream * (idxTo - idxFrom + 1));
    
		verify(nStream == linkedLayer->GetNumStreams());
    		verify(nUnroll == linkedLayer->GetNumUnrollSteps());

   		Matrix<INT> targetSubMat;
		Matrix<INT> gradientMaskSubMat;
		targetSubMat.Resize(1,nStream*(idxTo-idxFrom+1));
		gradientMaskSubMat.Resize(1,nStream*(idxTo-idxFrom+1));
		targetSubMat.SetEngine(this->engine); 
		gradientMaskSubMat.SetEngine(this->engine); 
		

		//Matrix<INT> targetSubMat(*ptrMat,0,0,0,ptrMat->GetNumCols()-1);		
		//Matrix<INT> gradientMaskSubMat(*ptrMat,1,1,0,ptrMat->GetNumCols()-1);	
		//targetSubMat.leadingDim = 1;	
		//gradientMaskSubMat.leadingDim = 1;

                engine->StreamSynchronize(stream);
		ptrMat->HostPull(stream);
		targetSubMat.HostPull(stream);
		gradientMaskSubMat.HostPull(stream);
                engine->StreamSynchronize(stream);

		INT *ptr = ptrMat->GetHostData();	
		INT *ptrTarget = targetSubMat.GetHostData();	
		INT *ptrMask = gradientMaskSubMat.GetHostData();	
    for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
    {
        for(long frameIdx = (long) idxFrom; frameIdx <= (long) idxTo; frameIdx++)
        {
            INT *framePtr = ptr + ((frameIdx - idxFrom) * nStream + streamIdx) * 2;
            INT *framePtrTarget = ptrTarget + ((frameIdx - idxFrom) * nStream + streamIdx) ;
            INT *framePtrMask = ptrMask + ((frameIdx - idxFrom) * nStream + streamIdx) ;
            	INT seq = framePtr[0];
            	INT sig = framePtr[1];
	
		framePtrTarget[0] = seq;	
		framePtrMask[0] = sig;	
            	//INT Target1 = framePtrTarget[0];
            	//INT Mask1 = framePtrMask[0];
		//printf("seq[%d]: %d, sig[%d]: %d, Target: %d, Mask: %d\n",(frameIdx-idxFrom)*nStream+streamIdx,seq,(frameIdx-idxFrom)*nStream+streamIdx+1,sig,Target1,Mask1);
	}
    }	
	targetSubMat.HostPush();
	gradientMaskSubMat.HostPush();
                engine->StreamSynchronize(stream);
		
		//printf("targetSubMat Rows: %d, Cols: %d, leadingDim: %d\n",targetSubMat.GetNumRows(),targetSubMat.GetNumCols(),targetSubMat.GetLeadingDim());

                engine->OneHotEncode(targetSubMat, targetSub, stream);
                engine->Set0or1(gradientMaskSubMat, gradientMaskSub, stream);
            }
            break;

        default:
            verify(false);
    }
    //printf("(char) End SetTarget\n");
    //fflush(stdout);
}


void MultiClassifProbeChar::ComputeErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    //printf("(char) Start ComputeErr\n");
    //fflush(stdout);
    verify(engine != NULL);
    //verify(linkedLayer->actType == ACT_SOFTMAX);

    Matrix<FLOAT> targetSub(target, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> errSub(err, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> actSub(GetActivation(), idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> gradientMaskSub(gradientMask, idxFrom * nStream, idxTo * nStream + nStream - 1);

    Matrix<FLOAT> lambdaMatSub(lambdaMat,idxFrom * nStream, idxTo*nStream+nStream -1);

    engine->MatSet(lambdaMatSub,this->lambda,stream); 
    
	/* err = target - act */
    engine->MatCopy(targetSub, errSub, stream);
    engine->MatAdd(actSub, errSub, (FLOAT) -1, stream);
    engine->MatElemMult(errSub, gradientMaskSub , errSub, stream);
    engine->MatElemMult(errSub, lambdaMatSub , errSub, stream);
    //printf("(char) End ComputeErr\n");
    //fflush(stdout);
}


void MultiClassifProbeChar::EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output)
{
    //printf("(char) start EvaluateOnHOst\n");
    //fflush(stdout);
#if 1
    unsigned long nPartialError;
    double sePartialSum, cePartialSum;
    unsigned long dim, nFrame;
    unsigned long nSample_partial;

    dim = output.GetNumRows();
    nFrame = output.GetNumCols();

    sePartialSum = 0.0;
    cePartialSum = 0.0;
    nPartialError = 0;
    nSample_partial = 0;
    //nSample += nFrame;

    switch(target.GetDataType())
    {
        case MultiTypeMatrix::DATATYPE_FLOAT:
            {
                FLOAT *t, *o;

                Matrix<FLOAT> *targetMat = reinterpret_cast<Matrix<FLOAT> *>(target.GetMatrix());


                verify(dim == targetMat->GetNumRows());
                verify(nFrame == targetMat->GetNumCols());

                PStream hostStream;

                engine->StreamCreate(hostStream, engine->GetHostLoc());

                targetMat->HostPull(hostStream);
                output.HostPull(hostStream);

                engine->StreamSynchronize(hostStream);
                engine->StreamDestroy(hostStream);

                t = targetMat->GetHostData();
                o = output.GetHostData();



                #ifdef FRACTAL_USE_OMP
                #pragma omp parallel for reduction(+:sePartialSum, cePartialSum, nPartialError)
                #endif
                for(unsigned long i = 0; i < nFrame; i++)
                {
                    unsigned long tMaxIdx = 0;
                    unsigned long oMaxIdx = 0;
                    FLOAT tMax = t[i * dim];
                    FLOAT oMax = o[i * dim];

                    for(unsigned long j = 0; j < dim; j++)
                    {
                        unsigned long idx = i * dim + j;

                        FLOAT tCur = t[idx];
                        FLOAT oCur = o[idx];

                        FLOAT err = oCur - tCur;
                        sePartialSum += err * err;

                        if(tCur > (FLOAT) 0)
                            cePartialSum -= tCur * std::log(oCur + (double) 1e-300);

                        if(oCur > oMax)
                        {
                            oMax = oCur;
                            oMaxIdx = j;
                        }

                        if(tCur > tMax)
                        {
                            tMax = tCur;
                            tMaxIdx = j;
                        }
                    }

                    if(tMaxIdx != oMaxIdx) nPartialError++;
                }
            }
            break;

        case MultiTypeMatrix::DATATYPE_INT:
            {
              
		  FLOAT *o;
                INT *t;
                INT *ptr;
                INT *m;
                Matrix<INT> *targetMat = reinterpret_cast<Matrix<INT> *>(target.GetMatrix());

   		Matrix<INT> targetSubMat;
		Matrix<INT> gradientMaskSubMat;
		
		targetSubMat.Resize(1,targetMat->GetNumCols());
		gradientMaskSubMat.Resize(1,targetMat->GetNumCols());
		targetSubMat.SetEngine(this->engine); 
		gradientMaskSubMat.SetEngine(this->engine); 
	
		//Matrix<INT> targetSubMat(*targetMat,0,0,0,targetMat->GetNumCols()-1);
		//targetSubMat.leadingDim = 1;	
		
		//printf("targetMat nRows: %d nCols: %d\n", targetMat->GetNumRows(), targetMat->GetNumCols());
		fflush(stdout);
                //verify(targetMat->GetNumRows() == 1);
                verify(nFrame == targetMat->GetNumCols());

                PStream hostStream;

                engine->StreamCreate(hostStream, engine->GetHostLoc());
		

                //targetMat->HostPull(hostStream);
		targetMat->HostPull(hostStream);
		targetSubMat.HostPull(hostStream);
		gradientMaskSubMat.HostPull(hostStream);

                output.HostPull(hostStream);

		engine->StreamSynchronize(hostStream);
                engine->StreamDestroy(hostStream);

                stream.engine->StreamSynchronize(stream);

                ptr = targetMat->GetHostData();
                t = targetSubMat.GetHostData();
                m = gradientMaskSubMat.GetHostData();
                o = output.GetHostData();

        for(long frameIdx = (long) 0; frameIdx < (long) nFrame; frameIdx++)
        {
            INT *framePtr = ptr + (frameIdx * 2);
            INT *framePtrTarget = t + frameIdx ;
            INT *framePtrMask = m + frameIdx;
            	INT seq = framePtr[0];
            	INT sig = framePtr[1];
	
		framePtrTarget[0] = seq;	
		framePtrMask[0] = sig;
                //if(sig == 1)nSample_partial++;	
            	//INT Target1 = framePtrTarget[0];
            	//INT Mask1 = framePtrMask[0];
		//printf("seq[%d]: %d, sig[%d]: %d, Target: %d, Mask: %d\n",frameIdx,seq,frameIdx+1,sig,seq,sig);
	}
                
		

                #ifdef FRACTAL_USE_OMP
                #pragma omp parallel for reduction(+:sePartialSum, cePartialSum, nPartialError, nSample_partial)
                #endif
                for(unsigned long i = 0; i < nFrame; i++)
                {
                    //printf("target[%d]: %d\n",i,t[i]);
		    //fflush();
		    if(m[i]==0) continue;
		    verify(t[i] >= (INT) 0 && t[i] < (INT) dim);

                    unsigned long oMaxIdx = 0;
                    FLOAT oMax = o[i * dim];

                    for(unsigned long j = 0; j < dim; j++)
                    {
                        unsigned long idx = i * dim + j;

                        FLOAT oCur = o[idx];

                        FLOAT err = oCur - (FLOAT)((INT) j == t[i]);
                        sePartialSum += err * err;

                        if(oCur > oMax)
                        {
                            oMax = oCur;
                            oMaxIdx = j;
                        }
                    }

                    if(t[i] != (INT) oMaxIdx) nPartialError++;
		    //printf("(char) i:%d maxIdx:%d  max value: %f  target: %d nPartialError: %d m: %d\n",i, oMaxIdx,o[i*dim + t[i]], t[i], nPartialError, m[i]);
                    cePartialSum -= std::log(o[i * dim + t[i]] + (double) 1e-300);
		    nSample_partial++;
                }
            }
            break;

        default:
            verify(false);
    }
		nSample += nSample_partial;
                //printf("nSample: %u\n",nSample);


    seSum += sePartialSum;
    ceSum += cePartialSum;
    nError += nPartialError;
                stream.engine->StreamSynchronize(stream);
#endif
    //printf("(char) end EvaluateOnHOst\n");
    //fflush(stdout);
}


void MultiClassifProbeChar::ResetStatistics()
{
    nSample = 0;
    nError = 0;
    seSum = 0.0;
    ceSum = 0.0;
}


const double MultiClassifProbeChar::GetLoss()
{
    return GetWordPerplexity();
}


void MultiClassifProbeChar::PrintStatistics(std::ostream &outStream)
{
    outStream << "MSE: " << GetMeanSquaredError()
        << "  ACE: " << GetAverageCrossEntropy()
        << "  FER: " << GetFrameErrorRate()
        << "  WordPPL: " << GetWordPerplexity()<<" nSample: "<<nSample;
}

const unsigned long MultiClassifProbeChar::GetnOOV()
{
    return 0;
}

const unsigned long MultiClassifProbeChar::GetnSample()
{
    return nSample;
}

const double MultiClassifProbeChar::GetMeanSquaredError()
{
    return seSum / nSample;
}

const double MultiClassifProbeChar::GetWordPerplexity()
{
    return ceSum;
}


const double MultiClassifProbeChar::GetAverageCrossEntropy()
{
    return ceSum / nSample;
}


const double MultiClassifProbeChar::GetFrameErrorRate()
{
    return (double) nError / nSample;
}


void MultiClassifProbeChar::SetEngine(Engine *engine)
{
    TrainableProbe::SetEngine(engine);

    this->engine = engine;

    target.SetEngine(engine);
    gradientMask.SetEngine(engine);
    lambdaMat.SetEngine(engine);
}


}

