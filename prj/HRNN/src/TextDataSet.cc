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


#include "TextDataSet.h"

#include <cstring>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <map>


using namespace fractal;

const unsigned long TextDataSet::CHANNEL_TEXT_INPUT = 0;
const unsigned long TextDataSet::CHANNEL_TEXT_OUTPUT = 1;
const unsigned long TextDataSet::CHANNEL_WORDS = 2;
//const unsigned long TextDataSet::CHANNEL_SIG_OOV = 2;
const unsigned long TextDataSet::CHANNEL_SIG_WORDBOUNDARY = 3;
const unsigned long TextDataSet::CHANNEL_SIG_NEWSEQ = 4;


TextDataSet::TextDataSet()
{
    nSeq = 0;

    mapper.resize(256);
    charSet.resize(256);

    for(int i = 0; i < 256; i++)
    {
        mapper[i] = i;
        charSet[i] = i;
    }
}


void TextDataSet::SetCharSet(const std::string &charSet)
{
    this->charSet.resize(charSet.size());

    for(int i = 0; i < 256; i++)
    {
        mapper[i] = -1;
    }

    for(int i = 0; i < (int) charSet.size(); i++)
    {
        verify(mapper[(unsigned char) charSet[i]] == -1);
        mapper[(unsigned char) charSet[i]] = i;
    }

    this->charSet = charSet;
}

#if 0
void TextDataSet::Split(TextDataSet &target, const double fraction)
{
    unsigned long i, j, k, tmp, n, nSrc, nTarget;

    std::vector<unsigned long> originalNFrame(nFrame);
    std::vector<std::vector<unsigned char>> originalText(text);

    n = text.size();

    nTarget = ((double) n * fraction + 0.5);
    verify(nTarget > 0);

    nSrc = n - nTarget;
    verify(nSrc > 0);

    std::vector<unsigned long> index(n);

    for(i = 0; i < n; i++)
    {
        index[i] = i;
    }

    /* Shuffle */
    for(i = 0; i < n - 1; i++)
    {
        j = i + (rand() % (n - i));

        tmp = index[i];
        index[i] = index[j];
        index[j] = tmp;
    }

    text.clear();
    text.shrink_to_fit();
    text.resize(nSrc);

    nFrame.clear();
    nFrame.shrink_to_fit();
    nFrame.resize(nSrc);

    target.text.clear();
    target.text.shrink_to_fit();
    target.text.resize(nTarget);

    target.nFrame.clear();
    target.nFrame.shrink_to_fit();
    target.nFrame.resize(nTarget);

    for(i = j = k = 0; i < n; i++)
    {
        if(index[i] < nTarget)
        {
            target.text[j] = originalText[i];
            target.nFrame[j] = originalNFrame[i];
            j++;
        }
        else
        {
            text[k] = originalText[i];
            nFrame[k] = originalNFrame[i];
            k++;
        }
    }

    nSeq = nSrc;
    target.nSeq = nTarget;
}
#endif

static bool byDescendingFrequency(const frequency_t& a, const frequency_t& b)
{ return a.second > b.second; }

struct isGTE // greater than or equal
{ 
		size_t inclusive_threshold;
		bool operator()(const frequency_t& record) const 
		{ return record.second >= inclusive_threshold; }
} over = {1};

void TextDataSet::SetWordLabels(TextDataSet &target)
{
	target.words_hash = words_hash;
	
	for (auto &it: target.words_hash)
			std::cout <<"words split: " <<it.second << "\t" << it.first << std::endl;
}

const unsigned long TextDataSet::ReadTextData(const std::string &filenames, size_t  wordsNum)
{
	std::map<std::string, size_t> tally;
	std::string s;
	over.inclusive_threshold = wordsNum;
    //for(unsigned long i = 0; i < filenames.size(); i++)
    {
		std::ifstream file;
        file.open(filenames, std::ios_base::in);

        verify(file.is_open() == true);

        while(file>>s)
        {
			tally[s]++;
        }

        file.close();
    }
	std::copy(tally.begin(),tally.end(),back_inserter(words));
	words_t::iterator begin = words.begin(),
			end = std::partition(begin, words.end(), over);
	std::sort(begin, end, &byDescendingFrequency);

	words.resize(std::distance(begin,end));
	//	words.shrink_to_fit();
		//for (words_t::const_iterator it=begin; it!=end; it++)
		//		std::cout<<"before: " << it->second << "\t" << it->first << std::endl;
	size_t i = 0;
	for (words_t::iterator it=begin; it!=end; it++)
	{
		it->second = i;
		i++;
	}
	for (words_t::const_iterator it=begin; it!=end; it++)
	{
			words_hash.insert(*it);
	}
//	for (auto &x: words_hash)
//	{
//			std::cout <<"after: " <<x.second << "\t" << x.first << std::endl;
//	}


    return words_hash.size() + 1; //words + OOV
}

const unsigned long TextDataSet::InsertWord(std::string insert)
{
	std::cout<<"before: "<<words_hash.size()<<std::endl;
	frequency_t input;
	input.first = insert;
	input.second = words_hash.size();
	words_hash.insert(input);	
	std::cout<<"after: "<<words_hash.size()<<std::endl;
	return words_hash.size() + 1;
}

const unsigned long TextDataSet::ReadTextData(const std::string &filenames)
{
    std::list<std::string> textList;
    std::ifstream file;
    std::string buf;
    size_t pos1, pos2;

    text.clear();

    //for(unsigned long i = 0; i < filenames.size(); i++)
    {
        file.open(filenames, std::ios_base::in);

        verify(file.is_open() == true);

        while(file.eof() == false)
        {
            std::getline(file, buf);
            verify(file.bad() == false);

            pos1 = buf.find_first_not_of(" \n\r");
            if(pos1 == std::string::npos) continue;

            pos2 = buf.find_last_not_of(" \n\r");

            if(pos2 >= pos1)
            {
                textList.push_back(" " + buf.substr(pos1, pos2 - pos1 + 1)+ " \n"); // Change "\n" to "\n "
		    }
            if((pos1 = buf.find_last_not_of(charSet)) != std::string::npos)
            {
                std::cerr << "Unregistered character '" << buf[pos1] << "'(" << (int) (unsigned char) buf[pos1] << ")" << std::endl;
                verify(mapper[(unsigned char) buf[pos1]] != -1);
            }
        }

        file.close();
    }

    nSeq = textList.size();


    size_t totalSize = 0;

    for(auto &tmpStr : textList)
    {
        totalSize += tmpStr.size();
    }

    nFrame.resize(nSeq);
    nFrame.shrink_to_fit();

    rawText.resize(totalSize);
    rawText.shrink_to_fit();

    text.resize(nSeq);
    text.shrink_to_fit();

    unsigned long i = 0;
    unsigned char *rawPtr = rawText.data();

    for(auto &tmpStr : textList)
    {
        nFrame[i] = tmpStr.size();
        text[i] = rawPtr;

        for(size_t pos = 0; pos < nFrame[i]; pos++)
        {
            text[i][pos] = tmpStr.c_str()[pos];
        }

        rawPtr += nFrame[i];
        i++;
    }

    return text.size();
}


const unsigned long TextDataSet::GetNumChannel() const
{
    return 5;
}


const fractal::ChannelInfo TextDataSet::GetChannelInfo(const unsigned long channelIdx) const
{
    ChannelInfo channelInfo;

    switch(channelIdx)
    {
        case CHANNEL_TEXT_INPUT:
            channelInfo.dataType = ChannelInfo::DATATYPE_INDEX;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = charSet.size();
            break;
        
	case CHANNEL_TEXT_OUTPUT:
            channelInfo.dataType = ChannelInfo::DATATYPE_SEQ;
            channelInfo.frameSize = 2;
            channelInfo.frameDim = charSet.size();
            break;
        
		case CHANNEL_WORDS:
            channelInfo.dataType = ChannelInfo::DATATYPE_SEQ;
            channelInfo.frameSize = 2;
            channelInfo.frameDim = words_hash.size() + 1;// words + OOV
            break;
/*
        case CHANNEL_SIG_OOV:
            channelInfo.dataType = ChannelInfo::DATATYPE_VECTOR;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 1;
			break;
*/
		case CHANNEL_SIG_WORDBOUNDARY:
            channelInfo.dataType = ChannelInfo::DATATYPE_VECTOR;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 1;
			break;

        case CHANNEL_SIG_NEWSEQ:
            channelInfo.dataType = ChannelInfo::DATATYPE_VECTOR;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 1;
            break;

        default:
            verify(false);
    }

    return channelInfo;
}


const unsigned long TextDataSet::GetNumSeq() const
{
    return nSeq;
}


const unsigned long TextDataSet::GetNumFrame(const unsigned long seqIdx) const
{
    verify(seqIdx < nSeq);

    return nFrame[seqIdx];
}


void TextDataSet::GetFrameData(const unsigned long seqIdx, const unsigned long channelIdx,
        const unsigned long frameIdx, void *const frame)
{
    verify(seqIdx < nSeq);
    verify(frameIdx < nFrame[seqIdx]);

    switch(channelIdx)
    {
        case CHANNEL_TEXT_INPUT:
            {
                int v = mapper[text[seqIdx][frameIdx]];
				//std::cout<<text[seqIdx][frameIdx];
                if(v == -1)
                {
                    std::cerr << "Unregistered character '" << (char) text[seqIdx][frameIdx] << "'(" << (int) text[seqIdx][frameIdx] << ")" << std::endl;
                    verify(v != -1);
                }
                *reinterpret_cast<INT *>(frame) = (INT) v;
            }
            break;
        
	case CHANNEL_TEXT_OUTPUT:
            {
                int v = mapper[text[seqIdx][frameIdx]];
				//std::cout<<text[seqIdx][frameIdx];
                if(v == -1)
                {
                    std::cerr << "Unregistered character '" << (char) text[seqIdx][frameIdx] << "'(" << (int) text[seqIdx][frameIdx] << ")" << std::endl;
                    verify(v != -1);
                }
                reinterpret_cast<INT *>(frame)[0] = (INT) v;

					size_t pos1, pos2;
					std::string currentWord;
					unsigned long i = 0;
					unsigned long j = 1;
					bool flag_prev = false;
					bool flag_next = false;
					while(1)
					{
                                                
                                                //if((frameIdx-i)==0){pos1 = frameIdx-i; flag_prev = true;}
                                                if(text[seqIdx][frameIdx-i] == ' ' && flag_prev == false)
                                                {
                                                    pos1 = frameIdx-i;
                                                    flag_prev = true;
                                                }

						if(text[seqIdx][frameIdx+j] == ' ' && flag_next == false)
						{
							pos2 = frameIdx+j;	
                                                        flag_next = true;
						}/*
						else if(text[seqIdx][frameIdx+i] == ' ')
						{
							pos2 = frameIdx+i;
							break;
						}*/
                                                if(flag_prev == 1 && flag_next == 1) break;
						
                                                i++; j++;
					}
					for(int i = pos1+1; i<pos2 ;i++)
					{
						currentWord += text[seqIdx][i];
					}
						//std::cout<<"text["<<seqIdx<<"]["<<frameIdx<<"]]:"<<text[seqIdx][frameIdx];
						//std::cout<<" currentMaskWord:"<<currentWord<<std::endl;
					auto cnt = words_hash.find(currentWord);
					int b = (cnt == words_hash.end());
					reinterpret_cast<INT *>(frame)[1] = (INT) b;
					//reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) b;
            }
            break;

        case CHANNEL_WORDS:
            {
                unsigned char c = text[seqIdx][frameIdx];
                int b = (c == ' ');// || c == '\n'); // remove "\n" since word-level LM needs \n as a vocabulary 
                reinterpret_cast<INT *>(frame)[1] = (INT) b;
					int v = 0;
					size_t pos1, pos2;
					std::string currentWord;
					unsigned long i = 0;
					bool flag = false;
					while(1)
					{
						if(text[seqIdx][frameIdx+i] == ' ' && flag == false )
						{
							pos1 = frameIdx+i;	
							flag = true;
						}
						else if(text[seqIdx][frameIdx+i] == ' ')
						{
							pos2 = frameIdx+i;
							break;
						}
						i++;
					}
					for(int i = pos1+1; i<pos2 ;i++)
					{
						currentWord += text[seqIdx][i];
					}
						
						//currentWord.assign(subcurrentSentence,pos2+1,pos2-pos1+1);
					
						//std::cout<<"========================="<<std::endl;
						//std::cout<<"text["<<seqIdx<<"]["<<frameIdx<<"]]:"<<text[seqIdx][frameIdx];
						//std::cout<<" currentWord:"<<currentWord<<std::endl;
						//std::cout<<"nFrmae: "<<nFrame[seqIdx]<<std::endl;
					auto cnt = words_hash.find(currentWord);
					if(cnt == words_hash.end())
					{
							v = words_hash.size(); 
					}
					else
					{
							v = cnt->second;
					}
					if(v > words_hash.size() || v < 0)
					{
						std::cerr<< "Unexpected error occured '"<< v<<"'"<<std::endl;
					}
					//std::cout<<"number:"<<v<<std::endl;
                reinterpret_cast<INT *>(frame)[0] = (INT) v;
            }
            break;
       /*
		case CHANNEL_SIG_OOV:
			{
					size_t pos1, pos2;
					std::string currentWord;
					unsigned long i = 0;
					bool flag = false;
					while(1)
					{
						if(text[seqIdx][frameIdx+i] == ' ' && flag == false )
						{
							pos1 = frameIdx+i;	
							flag = true;
						}
						else if(text[seqIdx][frameIdx+i] == ' ')
						{
							pos2 = frameIdx+i;
							break;
						}
						i++;
					}
					for(int i = pos1+1; i<pos2 ;i++)
					{
						currentWord += text[seqIdx][i];
					}
					auto cnt = words_hash.find(currentWord);
					int b = (cnt == words_hash.end());
					reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) b;

			}
			break;
*/
		case CHANNEL_SIG_WORDBOUNDARY:
            {
                unsigned char c = text[seqIdx][frameIdx];
                int b = (c == ' '); //|| c == '\n'); // remove "\n" since word-level LM needs \n as a vocabulary 
                reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) b;
            }
            break;

        case CHANNEL_SIG_NEWSEQ:
            reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) (frameIdx == 0);
            break;

        default:
            verify(false);
    }
}

