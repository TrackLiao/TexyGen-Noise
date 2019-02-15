#!/usr/bin/env python
# coding=utf-8
from nltk.translate.bleu_score import sentence_bleu
import sys
import numpy as np
import re
import codecs
import os
import math
import operator
import json
from random import randint
from nltk.translate.bleu_score import SmoothingFunction

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    text = re.sub(r'[^\w\s]', '', text)
    file.close()
    text = text.split('\n')
    del text[-1]
    result = []

    for item in text:
        temp = item.split()
        result.append(temp)

    return result



def calscore(cand_list, ref, c_num, name):
    avg_2_g = 0
    avg_3_g = 0
    avg_4_g = 0
    avg_5_g = 0
    for i in range(0,c_num):
        cand = cand_list[randint(0, len(cand_list))]
        if c_num == 1:
            print("candidate : ", cand)
        g_2, g_3, g_4, g_5 = get_score(cand, ref)
        avg_2_g += g_2
        avg_3_g += g_3
        avg_4_g += g_4
        avg_5_g += g_5

    
    avg_2_g = avg_2_g / float(c_num)
    avg_3_g = avg_3_g / float(c_num)
    avg_4_g = avg_4_g / float(c_num)
    avg_5_g = avg_5_g / float(c_num)

    
        
    result = [avg_2_g, avg_3_g, avg_4_g, avg_5_g, name]
    return result
        

    

def get_score(candidate, reference):

    g_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0, 0))
    g_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0, 0))
    g_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0, 0))
    g_5 = sentence_bleu(reference, candidate, weights=(0.2, 0.2, 0.2, 0.2, 0.2))

    return g_2, g_3, g_4, g_5

references = load_doc('data/noise/data.txt')
result = []

cand_num = 10
print("Bleu for swap")
candidate_list = load_doc('save/result/seqgan_swap.txt')
result.append(calscore(candidate_list, references, cand_num, 'swap'))

print("Bleu for reverse")
candidate_list = load_doc('save/result/seqgan_reverse.txt')
result.append(calscore(candidate_list, references, cand_num, 'reverse'))

print("Bleu for misword")
candidate_list = load_doc('save/result/seqgan_misword.txt')
result.append(calscore(candidate_list, references, cand_num, 'misword'))

print("Bleu for wtense")
candidate_list = load_doc('save/result/seqgan_wtense.txt')
result.append(calscore(candidate_list, references, cand_num, 'wtense'))

print("Bleu for typo")
candidate_list = load_doc('save/result/seqgan_typo.txt')
result.append(calscore(candidate_list, references, cand_num, 'typo'))

with open('Bleu_score.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)
