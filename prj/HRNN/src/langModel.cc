/*
   Copyright 2015-2016 Kyuyeon Hwang (kyuyeon.hwang@gmail.com)

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


#include <ctime>
#include <cstdio>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <iomanip>

#include <fractal/fractal.h>

#include "langModelNetwork.h"
#include "TextDataSet.h"

#define USE_WORD_CLOCK // use this for HLSTM-A or HLSTM-B
#define USE_WORD_CLOCK_SKIP // use this for HLSTM-B

using namespace fractal;


void printUsage(const char *argv0)
{
    std::cout << "Usage: " << argv0 << " [ -t <corpus path> | -g <starting sentence> ] <workspace path>" << std::endl;
}


int main(int argc, char *argv[])
{
    long numLayers = 2;
    long layerWidth = std::stoi(argv[5]);
    FLOAT lambda_w = std::stof(argv[6]);
    FLOAT lambda_c = std::stof(argv[7]);
	double dropoutRate = 0.5;

    Rnn rnn;
    Engine engine;
    unsigned long long randomSeed;
    bool train;

    struct timespec ts;
    unsigned long inputChannel, outputChannel, inputDim, outputDim;
    unsigned long outputChannelWords, outputDimWords;
    //std::string charSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"&:;-()[]/$%~=@ \n";
    //std::string charSet = "abcdefghijklmnopqrstuvwxyz\n";
    //std::string charSet = "atenoisrhludcmfpkgybw<>vN.'xj$-qz&0193#285\\764/* \n";
    //std::string charSet = "abcdefghijklmnopqrstuvwxyz0123456789<>N.'$-&#\\/* ";// \n sentence end token is always modeling by the wWLM.
    //std::string charSet = "abcdefghijklmnopqrstuvwxyz0123456789<>N.'$-&#\\/* \n";
    std::string charSet;


    charSet.resize(256);
    for(int i = 0; i < 256; i++)
    {
        charSet[i] = i;
    }

    rnn.SetEngine(&engine);
    //rnn.SetComputeLocs({1});

   // if(argc != 4)
   // {
   //     printUsage(argv[0]);
   //     exit(1);
   // }

    if(std::string(argv[1]) == "-t")
    {
        train = true;
    }
    else if(std::string(argv[1]) == "-g")
    {
        train = false;
    }
    else
    {
        printUsage(argv[0]);
        exit(1);
    }

    std::string workspacePath = argv[3];
    int surviveWordN = std::stoi(argv[4]);

    /* Set random seed */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    //randomSeed = ts.tv_nsec;
    randomSeed = 0;

    printf("Random seed: %lld\n\n", randomSeed);


    TextDataSet textTrainData, textDevData, textTestData;
    DataStream textTrainDataStream, textDevDataStream, textTestDataStream;

    textTrainData.SetCharSet(charSet);
    textDevData.SetCharSet(charSet);
    textTestData.SetCharSet(charSet);

    //if(train == true)
    {
        std::string corpusPath = argv[2];
        std::string trainFiles, devFiles, testFiles;

        std::cout << std::endl << "===== Training set =====" << std::endl;

        //for(int i = 0; i < 1; i++)
        {
            std::ostringstream tmpStr;

            tmpStr << corpusPath << "/ptb/train.txt";
            //tmpStr << corpusPath << "/wikiText2/wiki.train.tokens";
            //tmpStr << corpusPath << "/wikiText2/wiki.train.raw";
            //tmpStr << corpusPath << "/wikiText2/wiki.train.raw.lt25remove";
            //tmpStr << corpusPath << "/wikiText103/wiki.train.tokens";

            std::cout << tmpStr.str() << std::endl;
            trainFiles = tmpStr.str();
        }

        std::cout << std::endl << "===== Dev set =====" << std::endl;

        {
            std::ostringstream tmpStr;

            tmpStr << corpusPath << "/ptb/valid.txt";
            //tmpStr << corpusPath << "/wikiText2/wiki.valid.tokens";
            //tmpStr << corpusPath << "/wikiText2/wiki.valid.raw";
            //tmpStr << corpusPath << "/wikiText2/wiki.valid.raw.lt25remove";
            //tmpStr << corpusPath << "/wikiText103/wiki.valid.tokens";

            std::cout << tmpStr.str() << std::endl;
            devFiles = tmpStr.str();
        }

        std::cout << std::endl << "===== Test set =====" << std::endl;
        
		{
            std::ostringstream tmpStr;

            tmpStr << corpusPath << "/ptb/test.txt";
            //tmpStr << corpusPath << "/wikiText2/wiki.test.tokens";
            //tmpStr << corpusPath << "/wikiText2/wiki.test.raw";
            //tmpStr << corpusPath << "/wikiText2/wiki.test.raw.lt25remove";
            //tmpStr << corpusPath << "/wikiText103/wiki.test.tokens";

            std::cout << tmpStr.str() << std::endl;
            testFiles = tmpStr.str();
        }

        std::cout << std::endl;

        std::cout << "Reading the training set ..." << std::endl;
        textTrainData.ReadTextData(trainFiles);
        outputDimWords = textTrainData.ReadTextData(trainFiles,surviveWordN);
		//textTrainData.InsertWord("affljalfj");//<S> sentence boundary symbol
		textTrainData.InsertWord("\n");//<\S> sentence end symbol
		std::cout<<"Train_words: "<<outputDimWords<<std::endl;

		std::cout << "Reading the dev set ..." << std::endl;
        textDevData.ReadTextData(devFiles);
        textTrainData.SetWordLabels(textDevData);
		//std::cout<<"Dev_words: "<<word_num<<std::endl;

        std::cout << "Reading the test set ..." << std::endl;
        textTestData.ReadTextData(testFiles);
        textTrainData.SetWordLabels(textTestData);
		//std::cout<<"Test_words: "<<word_num<<std::endl;

        textTrainDataStream.LinkDataSet(&textTrainData);
        textDevDataStream.LinkDataSet(&textDevData);
        textTestDataStream.LinkDataSet(&textTestData);
    }

    inputChannel = TextDataSet::CHANNEL_TEXT_INPUT;
    outputChannel = TextDataSet::CHANNEL_TEXT_OUTPUT;
	outputChannelWords = TextDataSet::CHANNEL_WORDS;

    inputDim = textTrainData.GetChannelInfo(inputChannel).frameDim;
    outputDim = textTrainData.GetChannelInfo(outputChannel).frameDim;
    outputDimWords = textTrainData.GetChannelInfo(outputChannelWords).frameDim;
   
    verify(inputDim == outputDim);

    printf("Train: %ld sequences\n", textTrainData.GetNumSeq());
    printf("  Dev: %ld sequences\n", textDevData.GetNumSeq());
    printf(" Test: %ld sequences\n", textTestData.GetNumSeq());

    printf("\n");

    printf(" Input dim: %ld\n", inputDim);
    printf("Output dim: %ld\n", outputDim);
    printf("Output dim words: %ld\n", outputDimWords);

    printf("\n");

    /* Setting random seeds */
    engine.SetRandomSeed(randomSeed);
    textTrainDataStream.SetRandomSeed(randomSeed);
    textDevDataStream.SetRandomSeed(randomSeed);
    textTestDataStream.SetRandomSeed(randomSeed);

    /* Create a network */

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    #ifdef USE_WORD_CLOCK
    #ifdef USE_WORD_CLOCK_SKIP
    //CreateLangModelNetwork3(rnn, inputDim, numLayers, layerWidth, train, dropoutRate);
    CreateLangModelNetwork4(rnn, inputDim, outputDimWords, numLayers, layerWidth, train, dropoutRate);
    #else
    CreateLangModelNetwork2(rnn, inputDim, numLayers, layerWidth, train, dropoutRate);
    #endif
    #else
    CreateLangModelNetwork(rnn, inputDim, numLayers, layerWidth, train, dropoutRate);
    #endif

    rnn.PrintNetwork(std::cout);
    printf("Number of weights: %ld\n\n", rnn.GetNumWeights());

    #if 0
    /* Load a pretrained network */
    rnn.LoadState(workspacePath + "/net/best/");
    #endif

    if(train == true)
    {
        AutoOptimizer autoOptimizer;

        textTrainDataStream.SetNumStream(64);
        textDevDataStream.SetNumStream(128);
        //textTrainDataStream.SetNumStream(1);
        //textDevDataStream.SetNumStream(1);

        /* Set ports */
        InputProbe inputProbe;
        MultiClassifProbeWord outputWordProbe;
        MultiClassifProbeChar outputCharProbe;
       
	    outputWordProbe.SetOOVNum(outputDimWords-1);// PTB dataset.i
	    outputWordProbe.Setlambda(lambda_w);
	    outputCharProbe.Setlambda(lambda_c);
 
	    rnn.LinkProbe(inputProbe, "LM_INPUT");
        rnn.LinkProbe(outputCharProbe, "LM_OUTPUT");
        rnn.LinkProbe(outputWordProbe, "LM_WORD_OUTPUT");

        #ifdef USE_WORD_CLOCK
        InputProbe inputWordClockProbe;
        rnn.LinkProbe(inputWordClockProbe, "LM_WORD_CLOCK");
        //InputProbe inputOOVClockProbe;
        //rnn.LinkProbe(inputOOVClockProbe, "LM_OOV_CLOCK");
        #endif

        PortMapList inputPorts, outputPorts;

        inputPorts.push_back(PortMap(&inputProbe, inputChannel));
        outputPorts.push_back(PortMap(&outputWordProbe, outputChannelWords));
        outputPorts.push_back(PortMap(&outputCharProbe, outputChannel));

        #ifdef USE_WORD_CLOCK
        inputPorts.push_back(PortMap(&inputWordClockProbe, TextDataSet::CHANNEL_SIG_WORDBOUNDARY));
        //inputPorts.push_back(PortMap(&inputOOVClockProbe, TextDataSet::CHANNEL_SIG_OOV));
        #endif

        /* Training */
        {
            autoOptimizer.SetWorkspacePath(workspacePath);
            autoOptimizer.SetInitLearningRate(1e-5);
            autoOptimizer.SetMinLearningRate(5e-8);
            autoOptimizer.SetLearningRateDecayRate(0.5);
            autoOptimizer.SetMaxRetryCount(5);
            autoOptimizer.SetMomentum(0.9);
            autoOptimizer.SetWeightNoise(0.0);
            autoOptimizer.SetAdadelta(true);
            //autoOptimizer.SetRmsprop(true);
            autoOptimizer.SetRmsDecayRate(0.99);

            autoOptimizer.Optimize(rnn,
                    textTrainDataStream, textDevDataStream,
                    inputPorts, outputPorts,
                    5 * 1024 * 1024,  321 * 1024, 128, 64);
        }

        /* Evaluate the best network */

        textTrainDataStream.SetNumStream(256);
        textDevDataStream.SetNumStream(256);
        textTestDataStream.SetNumStream(256);

        textTrainDataStream.Reset();
        textDevDataStream.Reset();
        textTestDataStream.Reset();

        std::cout << "Best network:" << std::endl;

#if 1
        Evaluator evaluator;

        outputWordProbe.ResetStatistics();
        outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTrainDataStream, inputPorts, outputPorts, 10 * 1024 * 1024, 32);
        printf("(Word) Train :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        printf("(Char) Train :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
                outputCharProbe.GetAverageCrossEntropy(),
                outputCharProbe.GetFrameErrorRate(),
                outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	printf("Global PPL : %f\n",exp(((outputWordProbe.GetWordPerplexity()+outputCharProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());


        outputWordProbe.ResetStatistics();
        outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textDevDataStream, inputPorts, outputPorts, 1100 * 1024, 32);
        printf("(Word) Dev :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        printf("(Char) Dev :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
                outputCharProbe.GetAverageCrossEntropy(),
                outputCharProbe.GetFrameErrorRate(),
                outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	printf("Global PPL Dev : %f\n",exp(((outputWordProbe.GetWordPerplexity()+outputCharProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());
        

	outputWordProbe.ResetStatistics();
        outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTestDataStream, inputPorts, outputPorts,  1023 * 1024, 32);
        printf("(Word) Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        printf("(Char) Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
                outputCharProbe.GetAverageCrossEntropy(),
                outputCharProbe.GetFrameErrorRate(),
                outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	printf("Global PPL Test : %f\n",exp(((outputWordProbe.GetWordPerplexity()+outputCharProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());
/*
outputWordProbe.ResetStatistics();
        outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTestDataStream, inputPorts, outputPorts,  1223 * 1024, 32);
        printf("(Word) Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        printf("(Char) Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
                outputCharProbe.GetAverageCrossEntropy(),
                outputCharProbe.GetFrameErrorRate(),
                outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	printf("Global PPL Test : %f\n",exp(((outputWordProbe.GetWordPerplexity()+outputCharProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());
	*/
#else
        Evaluator evaluator;

        outputWordProbe.ResetStatistics();
        //outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTrainDataStream, inputPorts, outputPorts, 10 * 1024 * 1024, 32);
        printf("(Word) Train :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        //printf("(Char) Train :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
        //        outputCharProbe.GetAverageCrossEntropy(),
        //        outputCharProbe.GetFrameErrorRate(),
        //        outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	printf("Global PPL : %f\n",exp(((outputWordProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	//printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());


        outputWordProbe.ResetStatistics();
        //outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textDevDataStream, inputPorts, outputPorts, 10 * 1024 * 1024, 32);
        printf("(Word) Dev :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        //printf("(Char) Dev :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
        //        outputCharProbe.GetAverageCrossEntropy(),
        //        outputCharProbe.GetFrameErrorRate(),
        //        outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	//printf("Global PPL Dev : %f\n",exp(((outputWordProbe.GetWordPerplexity()+outputCharProbe.GetWordPerplexity()))/(outputWordProbe.GetnOOV()+outputWordProbe.GetnSample())));
	printf("Global PPL : %f\n",exp(((outputWordProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	//printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());
        

	outputWordProbe.ResetStatistics();
        //outputCharProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTestDataStream, inputPorts, outputPorts, 10 * 1024 * 1024, 32);
        printf("(Word) Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0), outputWordProbe.GetWordPerplexity());
        //printf("(Char) Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f WordPPL: %f\n", outputCharProbe.GetMeanSquaredError(),
        //        outputCharProbe.GetAverageCrossEntropy(),
        //        outputCharProbe.GetFrameErrorRate(),
        //        outputCharProbe.GetAverageCrossEntropy() / log(2.0), outputCharProbe.GetWordPerplexity());
	//printf("Global PPL Test : %f\n",exp(((outputWordProbe.GetWordPerplexity()+outputCharProbe.GetWordPerplexity()))/(outputWordProbe.GetnOOV()+outputWordProbe.GetnSample())));
	printf("Global PPL : %f\n",exp(((outputWordProbe.GetWordPerplexity()))/(outputWordProbe.GetnSample())));
	
	//printf("(Debug) Word nOOV: %u, Word nSample: %u, Char nSample: %u\n ",outputWordProbe.GetnOOV(),outputWordProbe.GetnSample(), outputCharProbe.GetnSample());
#endif
/*
        outputWordProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textDevDataStream, inputPorts, outputPorts, 10 * 1024 * 1024, 32);
        printf("  Dev :  MSE: %f  ACE: %f  FER: %f  BPC: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0));

        outputWordProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTestDataStream, inputPorts, outputPorts, 10 * 1024 * 1024, 32);
        printf(" Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f\n", outputWordProbe.GetMeanSquaredError(),
                outputWordProbe.GetAverageCrossEntropy(),
                outputWordProbe.GetFrameErrorRate(),
                outputWordProbe.GetAverageCrossEntropy() / log(2.0));
*/
    }
    else
    {
        rnn.LoadState(workspacePath + "/net/best/");

        /* ================================================================ */
        /*                  Get data from the RNN output                    */
        /* ================================================================ */

        unsigned long nUnroll = 32, nStream = 1;
        Matrix<FLOAT> matInput(inputDim, nStream);
        Matrix<FLOAT> matOutputChar(outputDim, nStream);
        Matrix<FLOAT> matOutputWord(outputDimWords, nStream);
        //Matrix<FLOAT> matOutputCharMask(outputDim, nStream);
        //Matrix<FLOAT> matOutputWordMask(outputDimWord, nStream);
        //InputProbe inputProbe;
        //OutputProbe outputProbe;
        InputProbe inputProbe;
        MultiClassifProbeWord outputWordProbe;
        MultiClassifProbeChar outputCharProbe;
        const std::string startString = "\n" + std::string(argv[6]);
        unsigned long szStartString;
        unsigned long maxOutputIdxWord = 0;
        unsigned long maxOutputIdxChar = 0;
        FLOAT maxValWord=0;


        /* Set engine */
        matInput.SetEngine(&engine);
        matOutputChar.SetEngine(&engine);
        matOutputWord.SetEngine(&engine);

        std::cout << startString;

        /* Link I/O probes */
        rnn.LinkProbe(inputProbe, "LM_INPUT");
        rnn.LinkProbe(outputCharProbe, "LM_OUTPUT");
        rnn.LinkProbe(outputWordProbe, "LM_WORD_OUTPUT");

        #ifdef USE_WORD_CLOCK
        Matrix<FLOAT> matInputWordClock(1, nStream);
        matInputWordClock.SetEngine(&engine);
        InputProbe inputWordClockProbe;
        rnn.LinkProbe(inputWordClockProbe, "LM_WORD_CLOCK");
        #endif

        /* Unroll the network nUnroll times and replicate it nStream times */
        rnn.SetBatchSize(nStream, nUnroll);


        /* Initialize the forward activations */
        rnn.InitForward(0, nUnroll - 1);

        szStartString = startString.size();
	
	int remain_char = 0;	
	int char_coin = 0;
	int word_coin = 0;
        std::string current_word;
        for(unsigned long j = 0; j < 4000; j++)
        {
            if(j < szStartString)
            {
                int v = textTrainData.Map(startString.c_str()[j]);
                verify(v != -1);

                for(unsigned long i = 0; i < inputDim; i++)
                {
                    matInput.GetHostData()[i] = (FLOAT) ((unsigned long) v == i);
                }
            }
            else if(maxOutputIdxWord == outputDimWords - 1)
            {
                for(unsigned long i = 0; i < inputDim; i++)
                {
                    matInput.GetHostData()[i] = (FLOAT) (maxOutputIdxChar == i);
                }
            }
	    else if(maxOutputIdxWord < outputDimWords - 1)
	    {
                for(unsigned long i = 0; i < inputDim; i++)
                {
                    matInput.GetHostData()[i] = (FLOAT) (textTrainData.Map(current_word[current_word.size()-remain_char]) == i);
                }
		
	    }

            /* Copy the input sequence to the RNN (asynchronous) */
            {
                Matrix<FLOAT> stateSub(inputProbe.GetState(), j % nUnroll, j % nUnroll);

                matInput.HostPush();
                engine.MatCopy(matInput, stateSub, inputProbe.GetPStream());
                inputProbe.EventRecord();
            }

            #ifdef USE_WORD_CLOCK
            int b1 = textTrainData.Map(' ');
            //int b2 = textTrainData.Map('\n');

            if(j < szStartString)
            {
                int v = textTrainData.Map(startString.c_str()[j]);

                verify(v != -1);

                matInputWordClock.GetHostData()[0] = (FLOAT) ((v == b1));// || (v == b2));
            }
            else if(maxOutputIdxWord == outputDimWords -1)
            {
                matInputWordClock.GetHostData()[0] = (FLOAT) (((int) maxOutputIdxChar == b1));// || ((int) maxOutputIdx == b2));
            }
            else if(maxOutputIdxWord<outputDimWords-1)
            {
                matInputWordClock.GetHostData()[0] = (FLOAT) ((current_word.size()-remain_char == current_word.size()-1));
            }
            /* Copy the input sequence to the RNN (asynchronous) */
            {
                Matrix<FLOAT> stateSub(inputWordClockProbe.GetState(), j % nUnroll, j % nUnroll);

                matInputWordClock.HostPush();
                engine.MatCopy(matInputWordClock, stateSub, inputWordClockProbe.GetPStream());
                inputWordClockProbe.EventRecord();
            }
            #endif

            /* Forward computation */
            /* Automatically scheduled to be executed after the copy event through the input probe */
            rnn.Forward(j % nUnroll, j % nUnroll);


            /* Copy the output sequence from the RNN to the GPU memory of matOutput */
            /* Automatically scheduled to be executed after finishing the forward activation */
            Matrix<FLOAT> actSubWord(outputWordProbe.GetActivation(), j % nUnroll, j % nUnroll);
            Matrix<FLOAT> actSubChar(outputCharProbe.GetActivation(), j % nUnroll, j % nUnroll);

            outputWordProbe.Wait();
            outputCharProbe.Wait();
            engine.MatCopy(actSubWord, matOutputWord, outputWordProbe.GetPStream());
            engine.MatCopy(actSubChar, matOutputChar, outputCharProbe.GetPStream());


            /* Copy the output matrix from the device (GPU) to the host (CPU) */
            matOutputWord.HostPull(outputWordProbe.GetPStream());
            matOutputChar.HostPull(outputCharProbe.GetPStream());
            outputWordProbe.EventRecord();
            outputCharProbe.EventRecord();


            /* Since the above operations are asynchronous, synchronization is required */
            outputWordProbe.EventSynchronize();
            outputCharProbe.EventSynchronize();

            if(j >= szStartString - 1)
            {
		FLOAT randVal = (FLOAT) rand() / RAND_MAX;
#if 0
                if(char_coin == 0 && word_coin == 0)
                {
                /* Find the maximum output for word */
                maxValWord = (FLOAT) 0;
                maxOutputIdxWord = 0;

                for(unsigned long i = 0; i < outputDimWords; i++)
                {
                        //std::cout<<"["<<i<<"]"<<": "<<matOutputWord.GetHostData()[i]<<std::endl;
                    if(matOutputWord.GetHostData()[i] > maxValWord)
                    {
                        maxValWord = matOutputWord.GetHostData()[i];
                        maxOutputIdxWord = i;
                    }
                }
                }
#else           
     
                if(char_coin == 0 && word_coin == 0)
                {
                for(unsigned long i = 0; i < outputDimWords; i++)
                {
                    FLOAT prob = matOutputWord.GetHostData()[i];

                    if(prob >= randVal)
                    {
                        maxOutputIdxWord = i;
                        break;
                    }
                    randVal -= prob;
                }
                }
#endif
		if(maxOutputIdxWord == outputDimWords-1)
		{
			/* Find the maximum output for charcter */
			if(char_coin == 0)
			{
#if 0
				FLOAT maxValChar = (FLOAT) 0;
				maxOutputIdxChar = 0;
				
				for(unsigned long i = 0; i < outputDim; i++)
				{
					if(matOutputChar.GetHostData()[i] > maxValChar)
					{
						maxValChar = matOutputChar.GetHostData()[i];
						maxOutputIdxChar = i;
					}
				}
#else				
				randVal = (FLOAT) rand() / RAND_MAX;
				for(unsigned long i = 0; i < outputDim; i++)
				{
					FLOAT prob = matOutputChar.GetHostData()[i];

					if(prob >= randVal)
					{
						maxOutputIdxChar = i;
						break;
					}
					randVal -= prob;
				}
#endif
				const unsigned char outChar = charSet.c_str()[maxOutputIdxChar];
				//std::cout<<"<";
				std::cout << outChar;
				//if(outChar == '\n') std::cout << std::endl;
				std::cout.flush();
				char_coin = 1;
			}
			else if(char_coin == 1)
			{
#if 0
				FLOAT maxValChar = (FLOAT) 0;
				maxOutputIdxChar = 0;
				
				for(unsigned long i = 0; i < outputDim; i++)
				{
					if(matOutputChar.GetHostData()[i] > maxValChar)
					{
						maxValChar = matOutputChar.GetHostData()[i];
						maxOutputIdxChar = i;
					}
				}
#else				
				randVal = (FLOAT) rand() / RAND_MAX;
				for(unsigned long i = 0; i < outputDim; i++)
				{
					FLOAT prob = matOutputChar.GetHostData()[i];

					if(prob >= randVal)
					{
						maxOutputIdxChar = i;
						break;
					}
					randVal -= prob;
				}
#endif
				const unsigned char outChar = charSet.c_str()[maxOutputIdxChar];
				if(outChar == ' ')
				{
					//std::cout<<"> ";
					char_coin = 0;
				}
				std::cout << outChar;
				std::cout.flush();
			}
		}
		else
		{
			if(word_coin == 0)
                        {
                            for(auto &x: textTrainData.words_hash)
                            {
                                if(x.second == maxOutputIdxWord)
                                {
                                    std::cout<<'('<<x.first<<')'<<' ';
                                    std::cout.flush();
                                    current_word = x.first + ' ';
                                    //current_word+=' ';
                                    remain_char = current_word.size();
                                    break;
                                }
                            }
                            word_coin = 1;
                        }
                        else if(word_coin == 1)
                        {
                            remain_char--;
                        }

                        if(remain_char == 1) word_coin = 0;
			
		}
                //const unsigned char outChar = charSet.c_str()[maxOutputIdx];
                //std::cout << outChar;
                //if(outChar == '\n') std::cout << std::endl;
                //std::cout.flush();
            }
        }

        std::cout << std::endl;


        /* Unlink the probes */
        inputProbe.UnlinkLayer();
        outputWordProbe.UnlinkLayer();
        outputCharProbe.UnlinkLayer();
    }


    rnn.SetEngine(NULL);

    return 0;
}


