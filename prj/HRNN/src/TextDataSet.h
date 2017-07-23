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


#ifndef __TEXTDATASET_H__
#define __TEXTDATASET_H__

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <fractal/fractal.h>

typedef std::pair<std::string, size_t> frequency_t;
typedef std::vector<frequency_t> words_t;
typedef std::unordered_map<std::string, size_t> word_hash_t;

class TextDataSet : public fractal::DataSet
{
public:

	static const unsigned long CHANNEL_TEXT_INPUT;
	static const unsigned long CHANNEL_TEXT_OUTPUT;
	static const unsigned long CHANNEL_WORDS;
	//static const unsigned long CHANNEL_SIG_OOV;
	static const unsigned long CHANNEL_SIG_WORDBOUNDARY;
	static const unsigned long CHANNEL_SIG_NEWSEQ;

    TextDataSet();

    //void Split(TextDataSet &target, const double fraction);
	void SetWordLabels(TextDataSet &target);
	const unsigned long InsertWord(std::string insert);

    void SetCharSet(const std::string &charSet);
    inline int Map(unsigned char c) { return mapper[c]; }

    const unsigned long ReadTextData(const std::string &filenames);
    const unsigned long ReadTextData(const std::string &filenames, size_t wordsNum);
	//static	bool byDescendingFrequency(const frequency_t& a, const frequency_t& b);

    const unsigned long GetNumChannel() const;
    const fractal::ChannelInfo GetChannelInfo(const unsigned long channelIdx) const;
    const unsigned long GetNumSeq() const;
    const unsigned long GetNumFrame(const unsigned long seqIdx) const;

    void GetFrameData(const unsigned long seqIdx, const unsigned long channelIdx,
            const unsigned long frameIdx, void *const frame);
word_hash_t words_hash;

protected:
    unsigned long nSeq;

    std::vector<unsigned long> nFrame;
    std::vector<int> mapper;
    std::string charSet;

    std::vector<unsigned char *> text;
    std::vector<unsigned char> rawText;

	//for word-level lang model
	words_t words;

};


#endif /* __TEXTDATASET_H__ */

