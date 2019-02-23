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
            ynum = round(float(row[y_type]), 10)
            x.append(xnum)
            y.append(ynum)
        return x , y


#graph for EmbSic
x_real, y_real = load_data('seqgan-real.csv', 1)
plt.plot(x_real,y_real, linestyle='-', marker='+' ,label='real')

x_swap, y_swap = load_data('seqgan-swap.csv', 1)
plt.plot(x_swap,y_swap, linestyle='-', marker='o' ,label='swap')

x_cross, y_cross = load_data('seqgan-crossover.csv', 1)
plt.plot(x_cross,y_cross, linestyle='-', marker='8' ,label='crossover')

x_reverse, y_reverse = load_data('seqgan-reverse.csv', 1)
plt.plot(x_reverse,y_reverse, linestyle='--', marker='v' ,label='reverse')

x_misword, y_misword = load_data('seqgan-misword.csv', 1)
plt.plot(x_misword,y_misword,linestyle='-.', marker='X' ,label='misword')

x_wtense, y_wtense = load_data('seqgan-wtense.csv', 1)
plt.plot(x_wtense,y_wtense,linestyle=':', marker='s' ,label='wtense')

x_wtense_n, y_wtense_n = load_data('seqgan-wtense-noun.csv', 1)
plt.plot(x_wtense_n,y_wtense_n,linestyle=':', marker='h' ,label='wtense-noun')

x_wtense_v, y_wtense_v = load_data('seqgan-wtense-verb.csv', 1)
plt.plot(x_wtense_v,y_wtense_v,linestyle=':', marker='H' ,label='wtense-verb')

x_typo, y_typo = load_data('seqgan-typo.csv', 1)
plt.plot(x_typo,y_typo,linestyle='-', marker='d' ,label='typo')

plt.xlabel('Epoches')
plt.ylabel('Embedding Similarity')
# plt.title("Noise Type")
plt.title("amazone_dataset")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('EmbSim.png')
plt.close()

#graph for nll-test
x_real, y_real = load_data('seqgan-real.csv', 2)
plt.plot(x_real,y_real, linestyle='-', marker='+' ,label='real')

x_swap, y_swap = load_data('seqgan-swap.csv', 2)
plt.plot(x_swap,y_swap, linestyle='-', marker='o' ,label='swap')

x_cross, y_cross = load_data('seqgan-crossover.csv', 2)
plt.plot(x_cross,y_cross, linestyle='-', marker='8' ,label='crossover')

x_reverse, y_reverse = load_data('seqgan-reverse.csv', 2)
plt.plot(x_reverse,y_reverse, linestyle='--', marker='v' ,label='reverse')

x_misword, y_misword = load_data('seqgan-misword.csv', 2)
plt.plot(x_misword,y_misword,linestyle='-.', marker='X' ,label='misword')

x_wtense, y_wtense = load_data('seqgan-wtense.csv', 2)
plt.plot(x_wtense,y_wtense,linestyle=':', marker='s' ,label='wtense')

x_wtense_n, y_wtense_n = load_data('seqgan-wtense-noun.csv', 2)
plt.plot(x_wtense_n,y_wtense_n,linestyle=':', marker='h' ,label='wtense-noun')

x_wtense_v, y_wtense_v = load_data('seqgan-wtense-verb.csv', 2)
plt.plot(x_wtense_v,y_wtense_v,linestyle=':', marker='H' ,label='wtense-verb')

x_typo, y_typo = load_data('seqgan-typo.csv', 2)
plt.plot(x_typo,y_typo,linestyle='-', marker='d' ,label='typo')

plt.xlabel('Epoches')
plt.ylabel('NLL-test loss')
plt.title("amazone_dataset")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('NLL-test.png')
plt.close()
