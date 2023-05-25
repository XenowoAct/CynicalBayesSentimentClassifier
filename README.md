# CynicalBayesSentimentClassifier

A new method for sentiment analysis that aims to remove the disadvantages of the Naive Bayes method by making predictions according to grams (sequences of words) of length n. (I have no idea if this has been done before).

## Prerequisites

- Python 3.x
- numpy
- nltk

## Installation

1. Clone the repository:

```
git clone https://github.com/XenowoAct/CynicalBayesSentimentClassifier
```

2. Install the required packages:

```
pip install nltk numpy
```

### Usage

1. Format your dataset into a labelled dictionary (As shown in line 126 of the code)

2. Use the `predictDoc` function to predict the sentiment of any document. The function takes the document you want to predict and the labelled dictionary as argument and returns a float number.
A possitive prediction indicates that the module predicted positive sentiment, and vice versa.

### Notes

Some clarifications on the arguments of the function `predictDoc` :

1. doc: A string to be predicted.
2. labelledTweets: A training dictionary in format {tweet:sentiment} where 0 means negative sentiment and 1 means positive.
3. n: The maximum length for grams (word sequences) to train on. Values >=2 are recommended.
4. minOcc: A value that will be used for filtering the gram dictionary. Any grams that have a number of occurences lower than this value will be deleted from the dictionary.
5. gramWeight: 1 by default. Affects how much impact a gram will make due to it's length. 1 gramWeight means that grams of length 3 will be thrice as impactful in the calculations.
6. removeStop: False by default. Set to True to remove stop words from the training set.
