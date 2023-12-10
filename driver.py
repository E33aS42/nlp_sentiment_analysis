from __future__ import division
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import os.path
import time
import re

begin = time.process_time()# used to check the computational time
print(begin)

# Those files are from the Movie Review Dataset: https://ai.stanford.edu/amaas/data/sentiment/ 
train_path = "aclImdb/train"
test_path = "aclImdb/test"

### Create list of words used for clean-up ###
stopwords = set()
for line in open("stopwords.en.txt", 'r'):
    stopwords.add(line[:-1])

stopwords2 = {'\'s', 'n\'t', '\'m', '\'re', '\' ', '/', '\'ll',
              '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'I', 'My', 'You', 'They', 'His', 'He', 'She', 'Their',
              'The', 'There', 'That', 'This ', 'Those', 'These', 'In', 'So',
              'A', 'An', 'It', 'Or', ' ', ''}

## Use nltk library for NLP:
# from nltk.corpus import stopwords
# 
# filtered_words = [word for word in word_list if word not in stopwords.words('english')]

### Data Preprocessing ###

def imdb_train_data_preprocess(inpath, name, outpath="./"):
    """
    This module is used to extract and combine text files under train_path directory into
    imdb_tr.csv. Each text file in train_path is stored as a row in imdb_tr.csv. And imdb_tr.csv has two
    columns: text and label
    """

    # Return a list containing the names of the entries in the directory given by path.
    posi = os.listdir(inpath + '/pos')  # path name to files of positive polarity
    nega = os.listdir(inpath + '/neg')  # path name to files of negative polarity

    #  create an empty panda dataframe that will contain the whole text files
    csv_file = pd.DataFrame(columns=['text', 'polarity'])

    # Create a set of words of the training data
    set_words = set()

    # positive polarity (not supposed to be known for testing file)
    for i in range(len(posi)):
        file = open(inpath + '/pos/' + posi[i])  # open current test file
        text = file.read()  # read current test file
        text1 = re.split('\W+', text)  # Split the string by the pattern occurrence
        # \W Matches any character which is not a word character.
        # + Causes the resulting RE to match 1 or more repetitions of the preceding RE.
        text2 = text1[:]  # create a deep copy text2 file to work on later while text1 must not change

        # clean-up #
        for word in text1:
            if word in stopwords:
                text2.remove(word)
            if word in stopwords2:
                text2.remove(word)

        # store words into set_words
        for w in text2:
            if w not in set_words:
                set_words.add(w)

        text = ""
        for word in text2:
            text += word + ' '
        csv_file.loc[i] = [text, 1]  # add line to csv file with current text file


        # negative polarity (not supposed to be known for testing file)
    for i in range(len(nega)):
        file = open(inpath + '/pos/' + posi[i])  # open current test file
        text = file.read()  # read current test file
        text1 = re.split('\W+', text)  # Split the string by the pattern occurrence
        # \W Matches any character which is not a word character.
        # + Causes the resulting RE to match 1 or more repetitions of the preceding RE.
        text2 = text1[:]  # create a deep copy text2 file to be worked on while text1 must not change

        # clean-up #
        for word in text1:
            if word in stopwords:
                text2.remove(word)
            if word in stopwords2:
                text2.remove(word)

        # store words into set_words
        for w in text2:
            if w not in set_words:
                set_words.add(w)

        text = ""
        for word in text2:
            text += word + ' '
        csv_file.loc[i + len(posi)] = [text, 0]  # add line to csv file with current text file


    # csv_file = csv_file.sample(frac=1).reset_index(drop=True)   # shuffle dataframe solution 1
    csv_file = shuffle(csv_file).reset_index(drop=True)         # shuffle dataframe solution 2
    csv_file.to_csv(outpath + name, sep=',')

    return set_words

def imdb_test_data_preprocess(inpath, name, set_words, outpath="./"):
    """
    This module is used to extract and combine text files under train_path directory into
    imdb_tr.csv. Each text file in train_path is stored as a row in imdb_tr.csv. And imdb_tr.csv has two
    columns, text and label
    """

    # Return a list containing the names of the entries in the directory given by path.
    posi = os.listdir(inpath + '/pos')  # path name to files of positive polarity
    nega = os.listdir(inpath + '/neg')  # path name to files of negative polarity

    #  create an empty panda dataframe that will contain the whole text files
    csv_file = pd.DataFrame(columns=['text', 'polarity'])

    # Create a set of words of the training data

    # positive polarity (not supposed to be known for testing file)
    for i in range(len(posi)):
        file = open(inpath + '/pos/' + posi[i])  # open current test file
        text = file.read()  # read current test file
        text1 = re.split('\W+', text)  # Split the string by the pattern occurrence
        # \W Matches any character which is not a word character.
        # + Causes the resulting RE to match 1 or more repetitions of the preceding RE.
        text2 = text1[:]    # create a deep copy text2 file to work on while text1 must not change

        # clean-up #
        for word in text1:
            if word in stopwords:
                text2.remove(word)
            if word in stopwords2:
                text2.remove(word)

        # we check if the words in text can be found among set_words, words collected from the training data
        # if there are not in set_words, they are deleted from text
        text3 = text2[:]     # create a deep copy text3 file to work on while text2 must not change
        for w in text3:
            if w not in set_words:
                text3.remove(w)

        text = ""
        for word in text3:
            text += word + ' '


        csv_file.loc[i] = [text, 1]  # add line to csv file with current text file


        # negative polarity (not supposed to be known for testing file)
    for i in range(len(nega)):
        file = open(inpath + '/pos/' + posi[i])  # open current test file
        text = file.read()  # read current test file
        text1 = re.split('\W+', text)  # Split the string by the pattern occurrence
        # \W Matches any character which is not a word character.
        # + Causes the resulting RE to match 1 or more repetitions of the preceding RE.
        text2 = text1[:]  # create a deep copy text2 file to be worked on while text1 must not change

        # clean-up #
        for word in text1:
            if word in stopwords:
                text2.remove(word)
            if word in stopwords2:
                text2.remove(word)

        # we check if the words in text can be found among set_words, words collected from the training data
        # if there are not in set_words, they are deleted from text
        text3 = text2[:]     # create a deep copy text3 file to work on while text2 must not change

        for w in text3:
            if w not in set_words:
                text3.remove(w)

        text = ""
        for word in text2:
            text += word + ' '

        csv_file.loc[i + len(posi)] = [text, 0]  # add line to csv file with current text file

    csv_file.to_csv(outpath + name, sep=',')


def prepare_data(train_data, test_data):
    # extract training data
    text_tr = train_data.iloc[:, 1].values  # extract text values
    polarity_tr = train_data.iloc[:, 2].values  # extract polarity values
    print('text_tr:%s' % text_tr.shape)
    print('polarity_tr:%s' % polarity_tr.shape)
    print(polarity_tr)
    polarity_tr = polarity_tr.astype(int)  # switch polarity data type from int64 to int32

    # extract testing data
    text_te = test_data.iloc[:, 1].values  # extract text values
    polarity_te = test_data.iloc[:, 2].values  # extract polarity values (not supposed to be known)
    print('text_te:%s' % text_te.shape)
    print('polarity_te:%s' % polarity_te.shape)
    polarity_te = polarity_te.astype(int)  # switch polarity data type from int64 to int32

    return text_tr, text_te, polarity_tr, polarity_te

### data representations models ###

## Unigram

def unigram(text_tr, text_te, polarity_tr, polarity_te):
    vect = CountVectorizer()            # Module used to convert a collection of raw documents to a term-document matrix
    vect.fit(text_tr)                   # calculate the fitting parameters
    Xtr = vect.transform(text_tr)       # apply the transformation using the fitting parameters
    print('Xtr:(%s, %s)'%(Xtr.shape))
    Xte = vect.transform(text_te)
    print('Xte:(%s, %s)'%(Xte.shape))
    clf = SGDClassifier(loss='hinge', penalty='l1') # apply the transformation using the fitting parameters to training and testing data
    clf.fit(Xtr, polarity_tr)           # Fit linear model with SGD
    prediction = clf.predict(Xte)       # predict labels of test data

    # comparing predictions to real polarity values
    percent = Test_prediction(prediction, polarity_te)
    print("percentage_Unigram: %s%%" %Test_prediction(prediction, polarity_te))

    # write predictions into output file
    out = open('unigram.output.txt','w')
    for i in range(len(prediction)):
        out.write('%s\n'%int(prediction[i]))    # save predicted labels into output file
    out.close()

    return percent, Xtr, Xte

## Bigram

def bigram(text_tr, text_te, polarity_tr, polarity_te):
    # Convert a collection of raw documents to a term-document matrix
    # ngram_range: boundary of the range of n-values for different n-grams to be extracted
    # Here we extract n-grams of size 2
    # bigram_vect = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b')
    bigram_vect = CountVectorizer(ngram_range=(2,2))
    bigram_vect.fit(text_tr)
    Xtr2 = bigram_vect.transform(text_tr)
    Xte2 = bigram_vect.transform(text_te)

    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtr2, polarity_tr)
    prediction = clf.predict(Xte2)

    # comparing predictions to real polarity values
    percent = Test_prediction(prediction, polarity_te)
    print("percentage_Bigram: %s%%" %Test_prediction(prediction, polarity_te))

    # write predictions into output file
    out = open('bigram.output.txt', 'w')
    for i in range(len(prediction)):
        out.write('%s\n' % int(prediction[i]))
    out.close()

    return percent, Xtr2, Xte2

## Unigram Tf-idf

def unigram_tdidf(Xtr, Xte, polarity_tr, polarity_te):
    trans = TfidfTransformer()              # Transform a count matrix to a normalized tf-idf representation
    # we fit and transform data from count matrices previously obtained in Unigram function
    Xtr_tfidf = trans.fit_transform(Xtr)
    Xte_tfidf = trans.transform(Xte)
    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtr_tfidf, polarity_tr)
    prediction = clf.predict(Xte_tfidf)

    # comparing predictions to real polarity values
    percent = Test_prediction(prediction, polarity_te)
    print("percentage_Unigram_Tfidf: %s%%" %Test_prediction(prediction, polarity_te))

    # write predictions into output file
    out = open('unigramtfidf.output.txt','w')
    for i in range(len(prediction)):
        out.write('%s\n'%int(prediction[i]))
    out.close()

    return percent, Xtr_tfidf, Xte_tfidf

## Bigram Tf-idf

def bigram_tdidf(text_tr, text_te, polarity_tr, polarity_te):
    tf_vect = TfidfVectorizer(ngram_range=(2, 2))   # Convert a collection of raw documents directly to a matrix of TF-IDF features
    Xtr2_tfidf = tf_vect.fit_transform(text_tr)
    Xte2_tfidf = tf_vect.transform(text_te)

    print('Xtr2_tfidf:(%s, %s)'%(Xtr2_tfidf.shape))
    print('Xte2_tfidf:(%s, %s)'%(Xte2_tfidf.shape))

    clf = SGDClassifier(loss='hinge', penalty='l1')
    clf.fit(Xtr2_tfidf, polarity_tr)
    prediction = clf.predict(Xte2_tfidf)

    # comparing predictions to real polarity values
    percent = Test_prediction(prediction, polarity_te)
    print("percentage_Bigram_Tfidf: %s%%" % percent)

    # write predictions into output file
    out = open('bigramtfidf.output.txt', 'w')
    for i in range(len(prediction)):
        out.write('%s\n'%int(prediction[i]))
    out.close()

    return percent, Xtr2_tfidf, Xte2_tfidf

def Test_prediction(prediction, polarity_te):
    # comparing predictions to real polarity values
    comp = 0
    for i in range(len(prediction)):
        if prediction[i] == polarity_te[i]:
            comp += 1                   # count number of polarities correctly predicted
    percent = comp/len(prediction)*100  # percentage of correct predictions
    print("count: %s ; prediction: %d" % (comp, len(prediction)))
    return percent

if __name__ == "__main__":
    # """train a SGD classifier using unigram representation,
    # predict sentiments on imdb_te.csv, and write output to
    # unigram.output.txt
    # train a SGD classifier using bigram representation,
    # predict sentiments on imdb_te.csv, and write output to
    # bigram.output.txt
    # train a SGD classifier using unigram representation
    # with tf-idf, predict sentiments on imdb_te.csv, and write
    # output to unigramtfidf.output.txt
    # train a SGD classifier using bigram representation
    # with tf-idf, predict sentiments on imdb_te.csv, and write
    # output to bigramtfidf.output.txt"""


    set_words = imdb_train_data_preprocess(train_path, "imdb_tr.csv")
    imdb_test_data_preprocess(test_path, "imdb_te.csv", set_words)
    print("train data preprocessed")
    train_data = pd.read_csv("imdb_tr.csv", encoding='latin-1')
    test_data = pd.read_csv("imdb_te.csv", encoding='latin-1')

    # set_words = imdb_train_data_preprocess(train_path, "imdb_tr_sam.csv")
    # imdb_test_data_preprocess(test_path, "imdb_te_sam.csv", set_words)
    # print("train data preprocessed")
    # train_data = pd.read_csv("imdb_tr_sam.csv", encoding='latin-1')
    # test_data = pd.read_csv("imdb_te_sam.csv", encoding='latin-1')


    text_tr, text_te, polarity_tr, polarity_te = prepare_data(train_data, test_data)

    P_Uni, Xtr, Xte = unigram(text_tr, text_te, polarity_tr, polarity_te)
    print("Unigram model done")
    P_Bi, Xtr2, Xte2 = bigram(text_tr, text_te, polarity_tr, polarity_te)
    print("Bigram model done")
    P_Uni_tfidf, Xtr_tfidf, Xte_tfidf = unigram_tdidf(Xtr, Xte, polarity_tr, polarity_te)
    print("Unigram tdidf model done")
    P_Bi_tfidf, Xtr2_tfidf, Xte2_tfidf = bigram_tdidf(text_tr, text_te, polarity_tr, polarity_te)
    print("Bigram tdidf model done")

    end = time.process_time()
    diff = end - begin
    print('time: %ss' % diff)