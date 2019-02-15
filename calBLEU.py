#!/usr/bin/env python
# coding=utf-8
from utils.metrics.Bleu import Bleu
real_file = 'data/noise/data.txt'

def print_bleu(test_file, real_file, name):
    result = []
    print("bleu for ", name)
    Bleu_2 = Bleu(test_text=test_file, real_text=real_file, gram=2)
    Bleu_2_score = Bleu_2.get_score()
    temp = "Bleu-2 : " + str(Bleu_2_score)
    print(temp)


    result.append(temp)

    Bleu_3 = Bleu(test_text=test_file, real_text=real_file, gram=3)
    Bleu_3_score = Bleu_3.get_score()
    temp = "Bleu-3 : " + str(Bleu_3_score)
    print(temp)
    result.append(temp)

    Bleu_4 = Bleu(test_text=test_file, real_text=real_file, gram=4)
    Bleu_4_score = Bleu_4.get_score()
    temp = "Bleu-4 : " + str(Bleu_4_score)
    print(temp)
    result.append(temp)

    Bleu_5 = Bleu(test_text=test_file, real_text=real_file, gram=5)
    Bleu_5_score = Bleu_5.get_score()
    temp = "Bleu-5 : " + str(Bleu_5_score)
    print(temp)
    result.append(temp)
    outfile = name + '-bleu.txt'
    with open(outfile, 'w') as f:
        for item in result:
            f.write("%s\n" % item)



test_file = 'save/result/seqgan_swap.txt'
print_bleu(test_file, real_file, 'swap')

test_file = 'save/result/seqgan_reverse.txt'
print_bleu(test_file, real_file, 'reverse')

test_file = 'save/result/seqgan_misword.txt'
print_bleu(test_file, real_file, 'misword')

test_file = 'save/result/seqgan_wtense.txt'
print_bleu(test_file, real_file, 'wtense')

test_file = 'save/result/seqgan_typo.txt'
print_bleu(test_file, real_file, 'typo')


print("all done!")
