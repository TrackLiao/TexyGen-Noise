#!/usr/bin/env python
# coding=utf-8
from utils.metrics.SelfBleu import SelfBleu

real_file = 'save/result/amazone_result/seqgan_real.txt'

def count_score(in_file, noise):
    score_2 = SelfBleu(test_text=in_file, gram = 2).get_score()
    score_3 = SelfBleu(test_text=in_file, gram = 3).get_score()
    score_4 = SelfBleu(test_text=in_file, gram = 4).get_score()
    score_5 = SelfBleu(test_text=in_file, gram = 5).get_score()
    temp = [score_2, score_3, score_4, score_5, noise]
    return temp

print("Running, please wait")
result = []
title = ['gram-2', 'gram-3', 'gram-4', 'noise']
result.append(title)

print("Calculating for real data")

infile = 'save/result/amazone_result/seqgan_real.txt'
result.append(count_score(infile, 'real'))

print("Calculating for swap noise")

infile = 'save/result/amazone_result/seqgan_swap.txt'
result.append(count_score(infile, 'swap'))

print("Calculating for reverse noise")
infile = 'save/result/amazone_result/seqgan_reverse.txt'
result.append(count_score(infile, 'reverse'))

print("Calculating for misword noise")
infile = 'save/result/amazone_result/seqgan_misword.txt'
result.append(count_score(infile, 'misword'))

print("Calculating for wtense noise")
infile = 'save/result/amazone_result/seqgan_wtense.txt'
result.append(count_score(infile, 'wtense'))

print("Calculating for noun noise")
infile = 'save/result/amazone_result/seqgan_wtense_noun.txt'
result.append(count_score(infile, 'noun'))

print("Calculating for verb noise")
infile = 'save/result/amazone_result/seqgan_wtense_verb.txt'
result.append(count_score(infile, 'verb'))

print("Calculating for crossover noise")
infile = 'save/result/amazone_result/seqgan_crossover.txt'
result.append(count_score(infile, 'crossover'))

print("Calculating for typo noise")

infile = 'save/result/amazone_result/seqgan_typo.txt'
result.append(count_score(infile, 'typo'))

with open('selfBleu.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)

    f.close()
print("all done!")
