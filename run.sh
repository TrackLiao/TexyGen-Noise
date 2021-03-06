#!/bin/bash
# python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/swap.txt
# python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/reverse.txt
# python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/misword.txt
# python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/wtense.txt
# python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/typo.txt
# echo '$1 = ' $1


if [ "$1" = "help" ]; then
    echo "bash run.py <argument>"
    echo "     argument list:"
    echo "          real"
    echo "          swap"
    echo "          reverse"
    echo "          misword"
    echo "          wtense"
    echo "          noun"
    echo "          verb"
    echo "          typo"
    echo "          crossover"
elif [ "$1" = "real" ]; then
    echo "testing with real word"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/data.txt

elif [ "$1" = "swap" ]; then
    echo "testing with swap word"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/swap.txt


elif [ "$1" = "reverse" ]; then
    echo "testing with reverse sentence"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/reverse.txt


elif [ "$1" = "misword" ]; then
    echo "testing with misword"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/misword.txt



elif [ "$1" = "wtense" ]; then
    echo "testing with wrong tense"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/wtense.txt

elif [ "$1" = "noun" ]; then
    echo "testing with wrong tense noun only"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/wtense_noun.txt

elif [ "$1" = "verb" ]; then
    echo "testing with wrong tense verb only"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/wtense_verb.txt

elif [ "$1" = "typo" ]; then
    echo "testing with typo"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/typo.txt

elif [ "$1" = "crossover" ]; then
    echo "testing with crossover"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/cross_over.txt

else
    echo "use  'bash run.sh help' for more instruction"
fi
