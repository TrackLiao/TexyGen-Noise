#!/usr/bin/env python
# coding=utf-8
from random import randint
import re

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    text = re.sub(r'[^\w\s]', '', text)
    file.close()
    text = text.split('\n')

    return text

def save_doc(text, filename):
    file = open(filename, 'w')

    for item in text:
        file.write("%s\n" % item)

    file.close()



def select_samples(text):
    samples = []
    #select 300 sentence
    for i in range(200):
        pos = randint(0, len(text)-1)
        samples.append(text[pos])
    

    return samples

print("Generating samples for real data")
save_doc(select_samples(load_doc('../seqgan_real.txt')) , 'real.sample' )

print("Generating samples for swap data")
save_doc(select_samples(load_doc('../seqgan_swap.txt')) , 'swap.sample' )

print("Generating samples for reverse data")
save_doc(select_samples(load_doc('../seqgan_reverse.txt')) , 'reverse.sample' )

print("Generating samples for misword data")
save_doc(select_samples(load_doc('../seqgan_misword.txt')) , 'misword.sample' )

print("Generating samples for wtense data")
save_doc(select_samples(load_doc('../seqgan_wtense.txt')) , 'wtense.sample' )

print("Generating samples for noun data")
save_doc(select_samples(load_doc('../seqgan_wtense_noun.txt')) , 'noun.sample' )

print("Generating samples for verb data")
save_doc(select_samples(load_doc('../seqgan_wtense_verb.txt')) , 'verb.sample' )

print("Generating samples for crossover data")
save_doc(select_samples(load_doc('../seqgan_crossover.txt')) , 'crossover.sample' )

print("Generating samples for typo data")
save_doc(select_samples(load_doc('../seqgan_typo.txt')) , 'typo.sample' )
