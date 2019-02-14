#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import csv
def load_data(filename, y_type):

    with open(filename,'r') as csvfile:
        x = []
        y = []
        plots = csv.reader(csvfile, delimiter=',')
        next(plots, None) #skip header
        for row in plots:
            xnum = round(float(row[0]), 4)
            ynum = round(float(row[y_type]), 4)
            x.append(xnum)
            y.append(ynum)
        return x , y


#graph for EmbSic
x_swap, y_swap = load_data('seqgan-swap.csv', 1)
plt.plot(x_swap,y_swap, label='swap')

x_reverse, y_reverse = load_data('seqgan-reverse.csv', 1)
plt.plot(x_reverse,y_reverse, label='reverse')

x_misword, y_misword = load_data('seqgan-misword.csv', 1)
plt.plot(x_misword,y_misword, label='misword')

x_wtense, y_wtense = load_data('seqgan-wtense.csv', 1)
plt.plot(x_wtense,y_wtense, label='wtense')

x_typo, y_typo = load_data('seqgan-typo.csv', 1)
plt.plot(x_typo,y_typo, label='typo')

plt.xlabel('Epoches')
plt.ylabel('EmbeddingSimilarity')
plt.title("Noise Type")
plt.legend()

# plt.show()
plt.savefig('EmbSim.png')

#graph for nll-test
x_swap, y_swap = load_data('seqgan-swap.csv', 2)
plt.plot(x_swap,y_swap, label='swap')

x_reverse, y_reverse = load_data('seqgan-reverse.csv', 2)
plt.plot(x_reverse,y_reverse, label='reverse')

x_misword, y_misword = load_data('seqgan-misword.csv', 2)
plt.plot(x_misword,y_misword, label='misword')

x_wtense, y_wtense = load_data('seqgan-wtense.csv', 2)
plt.plot(x_wtense,y_wtense, label='wtense')

x_typo, y_typo = load_data('seqgan-typo.csv', 2)
plt.plot(x_typo,y_typo, label='typo')

plt.xlabel('Epoches')
plt.ylabel('NLL-test')
plt.title("Noise Type")
plt.legend()

# plt.show()
plt.savefig('NLL-test.png')
