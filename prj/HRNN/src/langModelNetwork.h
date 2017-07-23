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


#ifndef __LANGMODELNETWORK_H__

#include <fractal/fractal.h>

void CreateLangModelNetwork(fractal::Rnn &rnn, const long numLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain = false, const double dropoutRate = 0.0);

void CreateLangModelNetwork2(fractal::Rnn &rnn, const long numLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain = false, const double dropoutRate = 0.0);

void CreateLangModelNetwork3(fractal::Rnn &rnn, const long numLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain = false, const double dropoutRate = 0.0);

void CreateLangModelNetwork4(fractal::Rnn &rnn, const long numLabels, const long numwordLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain = false, const double dropoutRate = 0.0);

void CreateLangModelNetwork5(fractal::Rnn &rnn, const long numLabels, const long numwordLabels,
        const long numLayers, const long layerWidth,
        const bool forTrain = false, const double dropoutRate = 0.0);
#endif /* __LANGMODELNETWORK_H__ */

