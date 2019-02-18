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
                if len(form) >= 1 :
                    lc = randint(0, len(form) - 1)
                    if randint(0,9) > 6:
                        word = form[lc]

            r.append(word)
        temp.append(r)
    out = np.asarray(temp)
    # print(out)
        
    return out

# generate noise sentence that change the tense (Noun Only)
def noise_tense_noun(text):
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

            r.append(word)
        temp.append(r)
    out = np.asarray(temp)
    # print(out)
        
    return out


# generate noise sentence that change the tense (Verb Only)
def noise_tense_verb(text):
    temp = []
    for i in range(0, text.shape[0]):
        r = []
        t = TextBlob(' '.join(text[i]))
        for word, pos in t.tags:
            if pos == 'VB':
                form = list(get_word_forms(word)['v'])
                if len(form) >= 2 :

                    # print("Verb : ", word)
                    lc = randint(0, len(form) - 1)
                    wordtemp = form[lc]
                    while (wordtemp == word) :
                        lc = randint(0, len(form) - 1)
                        wordtemp = form[lc]
                    word = wordtemp
                    # print("Change to : ", word)


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

# generate cross over text noise
def noise_cross_over(text):
    temp = []
    for i in range(0, text.shape[0]-1, 2):
        c_row = [] # current row
        n_row = [] # next row

        c_text = TextBlob(' '.join(text[i])) #current text
        n_text = TextBlob(' '.join(text[i+1])) #next text
        # initialize head and tail
        c_head = []
        c_tail = [] 
        n_head = [] 
        n_tail = []  
        c_cut = 0
        n_cut = 0
        
        if randint(0,9) > 2:
            for word, pos in c_text.tags:
                if pos != 'CC' and c_cut == 0:
                    c_head.append(word)
                elif pos == 'CC':
                    c_cut = 1
                if c_cut == 1 :
                    c_tail.append(word)

            for word, pos in n_text.tags:
                if pos != 'CC' and n_cut == 0:
                    n_head.append(word)
                elif pos == 'CC':
                    n_cut = 1
                if n_cut == 1 :
                    n_tail.append(word)

            c_row = [*c_head, *n_tail]
            n_row = [*n_head, *c_tail]


        else:
            for word, pos in c_text.tags:
                c_row.append(word)

            for word, pos in n_text.tags:
                n_row.append(word)


        temp.append(c_row)
        temp.append(n_row)
        


    out = np.asarray(temp)
    # print(out)
        
    return out

#load document
in_filename = 'data.txt'
# in_filename = 'image_coco.txt.bak'
doc = load_doc(in_filename)
doc = np.asarray(doc)
data = parse_array(doc)
data_bak = data.copy()



#generate text that have typo
typo_doc = noise_typo(data)
save_doc(typo_doc, 'typo.txt')
print("[1/5] typo.txt generated")




#generate reverse text
reverse_doc = noise_reverse(data)
# print(reverse_doc)
save_doc(reverse_doc, 'reverse.txt')
print("[2/5] reverse.txt generated")

#generate wrong tense text
wtense_doc = noise_tense(data)
save_doc(wtense_doc, 'wtense.txt')
print("[3/5] wtense.txt generated")



# generate wrong tense for noun only
#generate wrong tense text
wtense_noun_doc = noise_tense_noun(data)
save_doc(wtense_noun_doc, 'wtense_noun.txt')
print("[3.1/5] wtense_noun.txt generated")

# generate wrong tense for verb only
#generate wrong tense text
wtense_verb_doc = noise_tense_verb(data)
save_doc(wtense_verb_doc, 'wtense_verb.txt')
print("[3.2/5] wtense_verb.txt generated")



#generate text that remove some element
misword_doc = noise_misword(data)
save_doc(misword_doc, 'misword.txt')
print("[4/5] misword.txt generated")

# generate cross over
cross_over_doc = noise_cross_over(data)
# print("applying cross over")
# print(cross_over_doc)
#save_doc
save_doc(cross_over_doc, 'cross_over.txt')
print("[5.1/5] cross_over.txt generated")



#generate swap text
swap_doc = noise_swap(data_bak)
# print("random swap neighbour words")
# print(swap_doc)
#save_doc
save_doc(swap_doc, 'swap.txt')
print("[5.2/5] swap.txt generated")

