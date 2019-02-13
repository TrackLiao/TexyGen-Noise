#!/usr/bin/env python
# coding=utf-8
from numpy import array
import numpy as np
import re
import string
from random import randint
import random
from textblob import TextBlob
from word_forms.word_forms import get_word_forms
# from keras.preprocessing.text import Tokenizer

# load doc into memory

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    # text = file.readlines()
    text = file.read()
    text = re.sub(r'[^\w\s]','',text)
    # close the file
    file.close()
    text = text.split('\n')
    del text[-1]
    

    
    return text



# save doc to file
def save_doc(text, filename):
    temp = []
    for i in range(0, text.shape[0]):
        temp.append(' '.join(text[i]))
    file = open(filename, 'w')
    file.write('\n'.join(temp))
    file.close()

# convert 2D list to 2D array
def parse_array(text):
    
    arr = []
    for i in range(0, text.shape[0]):
        temp  = np.asarray(text[i].split())
        arr.append(temp)
    # convert data to np array
    arr = np.asarray(arr)
    return arr

# generate noise text that swap words
def noise_swap(text):
    temp = []
    for i in range(0, text.shape[0]):
        end = text[i].shape[0] - 1
        pos = randint(0, end)
        if pos == end :
            pos = pos - 1
        t = text[i][pos]
        text[i][pos] = text[i][pos+1]
        text[i][pos+1] = t

        temp.append(text[i])
    

    temp = np.asarray(temp)
    return temp

# generate noise sentence that reverse order
def noise_reverse(text):
    temp = []
    for i in range(0, text.shape[0]):
        temp.append(text[i][::-1])

    temp = np.asarray(temp)
    return temp


# generate noise sentence that change the tense
def noise_tense(text):
    temp = []
    for i in range(0, text.shape[0]):
        r = []
        t = TextBlob(' '.join(text[i]))
        for word, pos in t.tags:
            if pos == 'NN':
                if randint(0,9) > 6:
                    if word == word.lemmatize():
                        word = word.pluralize()
                    else:
                        word = word.lemmatize()
            if pos == 'VB':
                form = list(get_word_forms(word)['v'])
                if len(form) >= 2 :
                    lc = randint(0, len(form) - 1)
                    if randint(0,9) > 6:
                        word = form[lc]

            r.append(word)
        temp.append(r)
    out = np.asarray(temp)
    # print(out)
        
    return out


# generate noise sentence that delete some word
def noise_misword(text):
    temp = []
    # type of Noun, adjective, Verb, adverb, conjunction
    # candi = ['NN', 'JJ', 'VB', 'RB', 'CC']
    # type of Noun, adjective, Verb, adverb, conjunction
    candi = ['NN', 'VB', 'CC']
    for i in range(0, text.shape[0]):
        count = 0
        r = []
        t = TextBlob(' '.join(text[i]))
        c = randint(0,len(candi)-1)
        for word, pos in t.tags:
            if count <= 1:
                if pos == candi[c]:
                    count = count + 1
                else:
                    r.append(word)
            else:
                    r.append(word)
        temp.append(r)
    out = np.asarray(temp)
        
    return out

# generate noise sentence that contain typo
def noise_typo(text):
    temp = []
    for i in range(0, text.shape[0]):
        r = []
        t = TextBlob(' '.join(text[i]))
        for word, pos in t.tags:
            if randint(0,9) > 6:
                word = word + random.choice(string.ascii_letters)
                word = word.lower()

            r.append(word)
        temp.append(r)
    out = np.asarray(temp)
    # print(out)
        
    return out

#load document
in_filename = 'data.txt'
# in_filename = 'image_coco.txt.bak'
doc = load_doc(in_filename)
doc = np.asarray(doc)
data = parse_array(doc)





#generate swap text
swap_doc = noise_swap(data)
# print("random swap neighbour words")
# print(swap_doc)
#save_doc
save_doc(swap_doc, 'swap.txt')
print("[1/5] swap.txt generated")


#generate reverse text
reverse_doc = noise_reverse(data)
# print(reverse_doc)
save_doc(reverse_doc, 'reverse.txt')
print("[2/5] reverse.txt generated")

#generate wrong tense text
wtense_doc = noise_tense(data)
save_doc(wtense_doc, 'wtense.txt')
print("[3/5] wtense.txt generated")




#generate text that remove some element
misword_doc = noise_misword(data)
save_doc(misword_doc, 'misword.txt')
print("[4/5] misword.txt generated")


#generate text that have typo
typo_doc = noise_typo(data)
save_doc(typo_doc, 'typo.txt')
print("[5/5] typo.txt generated")

