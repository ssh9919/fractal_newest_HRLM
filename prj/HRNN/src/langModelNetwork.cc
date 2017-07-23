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


#include "langModelNetwork.h"


using namespace fractal;


void CreateLangModelNetwork(Rnn &rnn, const long numLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain, const double dropoutRate)
{
    InitWeightParamUniform initWeightParam;
    LayerParam dropoutLayerParam;

    initWeightParam.a = -sqrt(1.0 / layerWidth);
    initWeightParam.b = sqrt(1.0 / layerWidth);

    initWeightParam.isValid = forTrain;

    dropoutLayerParam.dropoutRate = dropoutRate;

    rnn.AddLayer("LM_INPUT", ACT_LINEAR, AGG_DONTCARE, numLabels);
    rnn.AddLayer("LM_OUTPUT", ACT_SOFTMAX, AGG_SUM, numLabels);


    for(int l = 0; l < numLayers; l++)
    {
        std::string layerIdx = std::to_string(l);
        std::string prevLayerIdx = std::to_string(l - 1);

        if(forTrain)
        {
            rnn.AddLayer("LSTM_LM[" + layerIdx + "].DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        }
        else
        {
            rnn.AddLayer("LSTM_LM[" + layerIdx + "].DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        }

        basicLayers::AddFastLstmLayer(rnn, "LSTM_LM[" + layerIdx + "]", "BIAS", 1, layerWidth, true, initWeightParam);


        if(l == 0)
        {
            rnn.AddConnection("LM_INPUT", "LSTM_LM[" + layerIdx + "].INPUT", initWeightParam);
        }
        else
        {
            rnn.AddConnection("LSTM_LM[" + prevLayerIdx + "].DROPOUT", "LSTM_LM[" + layerIdx + "].INPUT", initWeightParam);
        }

        if(l == numLayers - 1)
        {

            if(forTrain)
            {
                rnn.AddConnection("LSTM_LM[" + layerIdx + "].OUTPUT.DELAYED", "LSTM_LM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);
            }
            else
            {
                rnn.AddConnection("LSTM_LM[" + layerIdx + "].OUTPUT", "LSTM_LM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);
            }

            rnn.AddConnection("LSTM_LM[" + layerIdx + "].DROPOUT", "LM_OUTPUT", initWeightParam);
            rnn.AddConnection("BIAS", "LM_OUTPUT", initWeightParam);
        }
        else
        {
            rnn.AddConnection("LSTM_LM[" + layerIdx + "].OUTPUT", "LSTM_LM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);
        }
    }
}


void CreateLangModelNetwork2(Rnn &rnn, const long numLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain, const double dropoutRate)
{
    InitWeightParamUniform initWeightParam;
    LayerParam dropoutLayerParam;

    initWeightParam.a = -sqrt(1.0 / layerWidth);
    initWeightParam.b = sqrt(1.0 / layerWidth);

    initWeightParam.isValid = forTrain;

    dropoutLayerParam.dropoutRate = dropoutRate;


    /* Input and output layers */
    rnn.AddLayer("LM_INPUT", ACT_LINEAR, AGG_DONTCARE, numLabels);
    rnn.AddLayer("LM_WORD_CLOCK", ACT_LINEAR, AGG_DONTCARE, 1);
    rnn.AddLayer("LM_OUTPUT", ACT_SOFTMAX, AGG_SUM, numLabels);


    /* Internal layers */
    rnn.AddLayer("LM_CHAR_RESET", ACT_ONE_MINUS_LINEAR, AGG_SUM, 1);
    rnn.AddConnection("LM_WORD_CLOCK", "LM_CHAR_RESET", CONN_IDENTITY);

    basicLayers::AddFastLstmLayer(rnn, "LM_INPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);
    basicLayers::AddFastLstmLayer(rnn, "LM_OUTPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);

    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_INPUT_LSTM.INPUT", initWeightParam);

    rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    if(forTrain)
    {
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
    }
    else
    {
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
    }

    rnn.AddConnection("LM_INPUT", "LM_INPUT_LSTM.INPUT", initWeightParam);
    rnn.AddConnection("LM_INPUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.DELAYED.DROPOUT", CONN_IDENTITY);

    if(forTrain)
    {
        rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }
    else
    {
        rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }

    rnn.AddConnection("LM_OUTPUT_LSTM.DROPOUT", "LM_OUTPUT", initWeightParam);
    rnn.AddConnection("BIAS", "LM_OUTPUT", initWeightParam);


    for(int l = 0; l < numLayers; l++)
    {
        std::string layerIdx = std::to_string(l);
        std::string prevLayerIdx = std::to_string(l - 1);

        if(forTrain)
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        }
        else
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        }

        basicLayers::AddClockedLstmLayer(rnn, "LM_WORD_LSTM[" + layerIdx + "]", "BIAS", 1, layerWidth, true, initWeightParam);

        rnn.AddConnection("LM_WORD_CLOCK", "LM_WORD_LSTM[" + layerIdx + "].CLOCK", CONN_IDENTITY);

        if(l == 0)
        {
            rnn.AddConnection("LM_INPUT_LSTM.DELAYED.DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);

            /* Temporary. To distinguish ' ' and '\n' */
            rnn.AddConnection("LM_INPUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }
        else
        {
            rnn.AddConnection("LM_WORD_LSTM[" + prevLayerIdx + "].DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }


        rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].OUTPUT", "LM_WORD_LSTM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);

        if(l == numLayers - 1)
        {
            rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
        }
    }
}


void CreateLangModelNetwork3(Rnn &rnn, const long numLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain, const double dropoutRate)
{
    InitWeightParamUniform initWeightParam;
    LayerParam dropoutLayerParam;

    initWeightParam.a = -sqrt(1.0 / layerWidth);
    initWeightParam.b = sqrt(1.0 / layerWidth);

    initWeightParam.isValid = forTrain;

    dropoutLayerParam.dropoutRate = dropoutRate;


    /* Input and output layers */
    rnn.AddLayer("LM_INPUT", ACT_LINEAR, AGG_DONTCARE, numLabels);
    rnn.AddLayer("LM_WORD_CLOCK", ACT_LINEAR, AGG_DONTCARE, 1);
    rnn.AddLayer("LM_OUTPUT", ACT_SOFTMAX, AGG_SUM, numLabels);


    /* Internal layers */
    rnn.AddLayer("LM_CHAR_RESET", ACT_ONE_MINUS_LINEAR, AGG_SUM, 1);
    rnn.AddConnection("LM_WORD_CLOCK", "LM_CHAR_RESET", CONN_IDENTITY);

    basicLayers::AddFastLstmLayer(rnn, "LM_INPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);
    basicLayers::AddFastLstmLayer(rnn, "LM_OUTPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);

    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_INPUT_LSTM.INPUT", initWeightParam);

    rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    if(forTrain)
    {
        rnn.AddLayer("LM_INPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
    }
    else
    {
        rnn.AddLayer("LM_INPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
    }

    rnn.AddConnection("LM_INPUT", "LM_INPUT_LSTM.INPUT", initWeightParam);
    //rnn.AddConnection("LM_INPUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
    /* Skip connection */
    rnn.AddConnection("LM_INPUT_LSTM.DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.DROPOUT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.DELAYED.DROPOUT", CONN_IDENTITY);

    if(forTrain)
    {
        rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }
    else
    {
        rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }

    rnn.AddConnection("LM_OUTPUT_LSTM.DROPOUT", "LM_OUTPUT", initWeightParam);
    rnn.AddConnection("BIAS", "LM_OUTPUT", initWeightParam);


    for(int l = 0; l < numLayers; l++)
    {
        std::string layerIdx = std::to_string(l);
        std::string prevLayerIdx = std::to_string(l - 1);

        if(forTrain)
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        }
        else
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        }

        basicLayers::AddClockedLstmLayer(rnn, "LM_WORD_LSTM[" + layerIdx + "]", "BIAS", 1, layerWidth, true, initWeightParam);

        rnn.AddConnection("LM_WORD_CLOCK", "LM_WORD_LSTM[" + layerIdx + "].CLOCK", CONN_IDENTITY);

        if(l == 0)
        {
            rnn.AddConnection("LM_INPUT_LSTM.DELAYED.DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);

            /* Temporary. To distinguish ' ' and '\n' */
            rnn.AddConnection("LM_INPUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }
        else
        {
            rnn.AddConnection("LM_WORD_LSTM[" + prevLayerIdx + "].DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }


        rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].OUTPUT", "LM_WORD_LSTM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);

        if(l == numLayers - 1)
        {
            rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
        }
    }
}

void CreateLangModelNetwork4(Rnn &rnn, const long numLabels, const long numWordLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain, const double dropoutRate)
{
    InitWeightParamUniform initWeightParam;
    LayerParam dropoutLayerParam;

    initWeightParam.a = -sqrt(1.0 / layerWidth);
    initWeightParam.b = sqrt(1.0 / layerWidth);

    initWeightParam.isValid = forTrain;

    dropoutLayerParam.dropoutRate = dropoutRate;


    /* Input and output layers */
    rnn.AddLayer("LM_INPUT", ACT_LINEAR, AGG_DONTCARE, numLabels);
    rnn.AddLayer("LM_WORD_CLOCK", ACT_LINEAR, AGG_DONTCARE, 1);
    //rnn.AddLayer("LM_OOV_CLOCK", ACT_LINEAR, AGG_DONTCARE, 1);
    rnn.AddLayer("LM_OUTPUT", ACT_SOFTMAX, AGG_SUM, numLabels);
    rnn.AddLayer("LM_WORD_OUTPUT", ACT_SOFTMAX, AGG_SUM, numWordLabels);


    /* Internal layers */
    rnn.AddLayer("LM_CHAR_RESET", ACT_ONE_MINUS_LINEAR, AGG_SUM, 1);
    rnn.AddConnection("LM_WORD_CLOCK", "LM_CHAR_RESET", CONN_IDENTITY);

    basicLayers::AddFastLstmLayer(rnn, "LM_INPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);
    basicLayers::AddFastLstmLayer(rnn, "LM_OUTPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);

    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_INPUT_LSTM.INPUT", initWeightParam);

    rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    if(forTrain)
    {
        rnn.AddLayer("LM_INPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
    }
    else
    {
        rnn.AddLayer("LM_INPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
    }

    rnn.AddConnection("LM_INPUT", "LM_INPUT_LSTM.INPUT", initWeightParam);
    //rnn.AddConnection("LM_INPUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
    /* Skip connection */
    rnn.AddConnection("LM_INPUT_LSTM.DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.DROPOUT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.DELAYED.DROPOUT", CONN_IDENTITY);

    if(forTrain)
    {
        rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }
    else
    {
        rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }

    rnn.AddConnection("LM_OUTPUT_LSTM.DROPOUT", "LM_OUTPUT", initWeightParam);
    rnn.AddConnection("BIAS", "LM_OUTPUT", initWeightParam);


    for(int l = 0; l < numLayers; l++)
    {
        std::string layerIdx = std::to_string(l);
        std::string prevLayerIdx = std::to_string(l - 1);

        if(forTrain)
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        }
        else
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        }

        basicLayers::AddClockedLstmLayer(rnn, "LM_WORD_LSTM[" + layerIdx + "]", "BIAS", 1, layerWidth, true, initWeightParam);

        rnn.AddConnection("LM_WORD_CLOCK", "LM_WORD_LSTM[" + layerIdx + "].CLOCK", CONN_IDENTITY);

        if(l == 0)
        {
            rnn.AddConnection("LM_INPUT_LSTM.DELAYED.DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);

            /* Temporary. To distinguish ' ' and '\n' */
            rnn.AddConnection("LM_INPUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }
        else
        {
            rnn.AddConnection("LM_WORD_LSTM[" + prevLayerIdx + "].DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }


        rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].OUTPUT", "LM_WORD_LSTM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);

        if(l == numLayers - 1)
        {
            rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
            rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", "LM_WORD_OUTPUT", initWeightParam);
	    rnn.AddConnection("BIAS", "LM_WORD_OUTPUT", initWeightParam);
        }
    }
}
void CreateLangModelNetwork5(Rnn &rnn, const long numLabels, const long numWordLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain, const double dropoutRate)
{
    InitWeightParamUniform initWeightParam;
    LayerParam dropoutLayerParam;

    initWeightParam.a = -sqrt(1.0 / layerWidth);
    initWeightParam.b = sqrt(1.0 / layerWidth);

    initWeightParam.isValid = forTrain;

    dropoutLayerParam.dropoutRate = dropoutRate;


    /* Input and output layers */
    rnn.AddLayer("LM_INPUT", ACT_LINEAR, AGG_DONTCARE, numLabels);
    rnn.AddLayer("LM_WORD_CLOCK", ACT_LINEAR, AGG_DONTCARE, 1);
    //rnn.AddLayer("LM_OOV_CLOCK", ACT_LINEAR, AGG_DONTCARE, 1);
    //rnn.AddLayer("LM_OUTPUT", ACT_SOFTMAX, AGG_SUM, numLabels);
    rnn.AddLayer("LM_WORD_OUTPUT", ACT_SOFTMAX, AGG_SUM, numWordLabels);


    /* Internal layers */
    rnn.AddLayer("LM_CHAR_RESET", ACT_ONE_MINUS_LINEAR, AGG_SUM, 1);
    rnn.AddConnection("LM_WORD_CLOCK", "LM_CHAR_RESET", CONN_IDENTITY);

    basicLayers::AddFastLstmLayer(rnn, "LM_INPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);
    //basicLayers::AddFastLstmLayer(rnn, "LM_OUTPUT_LSTM", "BIAS", 1, layerWidth, false, initWeightParam);

    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    rnn.AddLayer("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_INPUT_LSTM.INPUT", initWeightParam);

    //rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED", ACT_LINEAR, AGG_SUM, layerWidth);
    //rnn.AddLayer("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", ACT_LINEAR, AGG_MULT, layerWidth);

    //rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.OUTPUT.DELAYED", {CONN_IDENTITY, 1});
    //rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_IDENTITY);
    //rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    if(forTrain)
    {
        //rnn.AddLayer("LM_INPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        ///rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
    }
    else
    {
        //rnn.AddLayer("LM_INPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        rnn.AddLayer("LM_INPUT_LSTM.DELAYED.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        //rnn.AddLayer("LM_OUTPUT_LSTM.DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
    }

    rnn.AddConnection("LM_INPUT", "LM_INPUT_LSTM.INPUT", initWeightParam);
    //rnn.AddConnection("LM_INPUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
    /* Skip connection */
    //rnn.AddConnection("LM_INPUT_LSTM.DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);

    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    rnn.AddConnection("LM_CHAR_RESET", "LM_INPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    //rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.OUTPUT.DELAYED_MULT", CONN_BROADCAST);
    //rnn.AddConnection("LM_CHAR_RESET", "LM_OUTPUT_LSTM.MEMORY_CELL.DELAYED", CONN_BROADCAST);

    //rnn.AddConnection("LM_INPUT_LSTM.OUTPUT", "LM_INPUT_LSTM.DROPOUT", CONN_IDENTITY);
    rnn.AddConnection("LM_INPUT_LSTM.OUTPUT.DELAYED", "LM_INPUT_LSTM.DELAYED.DROPOUT", CONN_IDENTITY);

    if(forTrain)
    {
        //rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT.DELAYED", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }
    else
    {
        //rnn.AddConnection("LM_OUTPUT_LSTM.OUTPUT", "LM_OUTPUT_LSTM.DROPOUT", CONN_IDENTITY);
    }

    //rnn.AddConnection("LM_OUTPUT_LSTM.DROPOUT", "LM_OUTPUT", initWeightParam);
    //rnn.AddConnection("BIAS", "LM_OUTPUT", initWeightParam);


    for(int l = 0; l < numLayers; l++)
    {
        std::string layerIdx = std::to_string(l);
        std::string prevLayerIdx = std::to_string(l - 1);

        if(forTrain)
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_DROPOUT, AGG_SUM, layerWidth, dropoutLayerParam);
        }
        else
        {
            rnn.AddLayer("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", ACT_LINEAR, AGG_SUM, layerWidth);
        }

        basicLayers::AddClockedLstmLayer(rnn, "LM_WORD_LSTM[" + layerIdx + "]", "BIAS", 1, layerWidth, true, initWeightParam);

        rnn.AddConnection("LM_WORD_CLOCK", "LM_WORD_LSTM[" + layerIdx + "].CLOCK", CONN_IDENTITY);

        if(l == 0)
        {
            rnn.AddConnection("LM_INPUT_LSTM.DELAYED.DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);

            /* Temporary. To distinguish ' ' and '\n' */
            rnn.AddConnection("LM_INPUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }
        else
        {
            rnn.AddConnection("LM_WORD_LSTM[" + prevLayerIdx + "].DROPOUT", "LM_WORD_LSTM[" + layerIdx + "].INPUT", initWeightParam);
        }


        rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].OUTPUT", "LM_WORD_LSTM[" + layerIdx + "].DROPOUT", CONN_IDENTITY);

        if(l == numLayers - 1)
        {
            //rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", "LM_OUTPUT_LSTM.INPUT", initWeightParam);
            rnn.AddConnection("LM_WORD_LSTM[" + layerIdx + "].DROPOUT", "LM_WORD_OUTPUT", initWeightParam);
	    rnn.AddConnection("BIAS", "LM_WORD_OUTPUT", initWeightParam);
        }
    }
}
