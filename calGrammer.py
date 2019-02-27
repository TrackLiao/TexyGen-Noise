#!/usr/bin/env python
# coding=utf-8
import re
import language_check

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    text = re.sub(r'[^\w\s]', '', text)
    file.close()
    text = text.split('\n')
    return text

def check_grammar(filename, noise):
    result = 0
    tool = language_check.LanguageTool('en-US')
    text = load_doc(filename)
    for item in text:
        result = result + len(tool.check(item))
    avg_result = result / len(text)

    out = [result, avg_result ,noise]
    return out

result = []
title = ['error', 'avg error' ,'noise']
result.append(title)

print("Calculating for real data")
infile = 'save/result/amazone_result/seqgan_real.txt'
result.append(check_grammar(infile, 'real'))

print("Calculating for swap data")
infile = 'save/result/amazone_result/seqgan_swap.txt'
result.append(check_grammar(infile, 'swap'))

print("Calculating for reverse data")
infile = 'save/result/amazone_result/seqgan_reverse.txt'
result.append(check_grammar(infile, 'reverse'))

print("Calculating for misword data")
infile = 'save/result/amazone_result/seqgan_misword.txt'
result.append(check_grammar(infile, 'misword'))

print("Calculating for wtense data")
infile = 'save/result/amazone_result/seqgan_wtense.txt'
result.append(check_grammar(infile, 'wtense'))

print("Calculating for noun data")
infile = 'save/result/amazone_result/seqgan_wtense_noun.txt'
result.append(check_grammar(infile, 'noun'))

print("Calculating for verb data")
infile = 'save/result/amazone_result/seqgan_wtense_verb.txt'
result.append(check_grammar(infile, 'verb'))

print("Calculating for crossover data")
infile = 'save/result/amazone_result/seqgan_crossover.txt'
result.append(check_grammar(infile, 'crossover'))

print("Calculating for typo data")
infile = 'save/result/amazone_result/seqgan_typo.txt'
result.append(check_grammar(infile, 'typo'))

with open('grammar_error.csv', 'w') as f:
    for item in result:
        f.write("%s\n" % item)

    f.close()

print("all done !!!")
