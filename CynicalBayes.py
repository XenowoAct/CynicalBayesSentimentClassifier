#!/usr/bin/python
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')






def sentencize(text):
    text = text.replace("n't",' not')
    text = text.replace("'ll",' will')
    text = text.replace("'m",' am')
    sentences = nltk.sent_tokenize(text)
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
        clean_sentences.append(clean_sentence)
    
    return clean_sentences

def remove_urls(text):
    url_pattern = re.compile(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?')
    return url_pattern.sub('', text)

def preprocess_tweet(tweet,removeStop):
    tweet = tweet.lower()
    tweet = remove_urls(tweet)
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z]", " ", tweet)
    tokens = word_tokenize(tweet)
    if removeStop:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if not token in stop_words]
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def generate_n_grams(sent,n):
    n_grams = []
    for i in range(len(sent) - n + 1):
        n_gram = sent[i : i+n]
        n_grams.append(n_gram)
    return n_grams

def labelGram(gram,sentiment,freqDict):
    gram = tuple(gram)
    if gram in freqDict.keys():
        freqDict[gram][sentiment] += 1 
    else:
        freqDict[gram] = [0,0]
        freqDict[gram][sentiment] = 1
    return freqDict

def addTweetToGramDict(n,tweet,sentiment,freqDict,removeStop):
    ngrams = []
    tweet = sentencize(tweet)
    tweet = [preprocess_tweet(_,removeStop) for _ in tweet]
    for n in range(1,n+1):
        ngrams += [generate_n_grams(_,n) for _ in tweet]
    for sentence in ngrams:
        for gram in sentence:
            # print(gram)
            freqDict = labelGram(gram,sentiment,freqDict)
    return freqDict

def generateGramDict(tweets, removeStop,n):
    freqdict = {}
    for tweet, label in tweets.items():
        freqdict = addTweetToGramDict(n,tweet,label,freqdict, removeStop)
    freqdict[('<unk>',)] = [0,0]
    return freqdict

def filterGramDict(freqDict,minOccurnce):
    newdict = freqDict.copy()
    for key, value in freqDict.items():
        if sum(value) < minOccurnce and key != ('<unk>',):
            newdict.pop(key)
    return newdict

def getV(freqDict):
    vocab_size = len(freqDict)
    # print(f'V = {vocab_size}')
    return vocab_size

def getN(sentiment, freqDict):
    count = sum(value[sentiment] for value in freqDict.values())
    # print(f'N = {count}')
    return count

def getGramProbability(freqDict, gram, sentiment,gramWeight):
    frequency = freqDict[gram][sentiment]
    return (frequency*len(gram)*gramWeight + 1) / (getN(sentiment, freqDict) + getV(freqDict))

def getGramLog(gram, freqDict, gramWeight):
    return np.log(getGramProbability(freqDict, gram, 1, gramWeight) / getGramProbability(freqDict, gram, 0, gramWeight))

def GetLogPrior(labelledTweets):
    labels = labelledTweets.values()
    Dneg = sum([1 for _ in labels if _ == 0])
    Dpos = sum([1 for _ in labels if _ == 1])
    return np.log(Dpos) - np.log(Dneg)

def predictDoc(doc,labelledTweets,n,minOcc,gramWeight=1,removeStop=False):
    freqdict = generateGramDict(labelledTweets,removeStop,n)
    freqdict = filterGramDict(freqdict,minOcc)
    tokens = list(generateGramDict({doc:0},removeStop,n).keys())
    print(tokens)
    print(freqdict)
    prediction = GetLogPrior(labelledTweets) + sum([getGramLog(token, freqdict, gramWeight) if token in freqdict else getGramLog(('<unk>',), freqdict, gramWeight) for token in tokens])
    return prediction


#Example Set
tweets = {'hey there, I love you bino!':1,'Oh, I actually fucking hate bino ngl':0,'Freedom is the best thing ever!':1,"You all should kill yourselves NOW":0,'Hey there, fuck you!':0,'that movie was not good at all!':0,'that was a good movie!':1}

print(predictDoc('bino is good! I love bino',tweets,3,0))


