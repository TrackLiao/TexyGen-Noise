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
    echo "          swap"
    echo "          reverse"
    echo "          misword"
    echo "          wtense"
    echo "          typo"
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


elif [ "$1" = "typo" ]; then
    echo "testing with typo"
    python main.py -g seqgan -t real -d ./data/noise/data.txt -f ./data/noise/typo.txt

else
    echo "use  'bash run.sh help' for more instruction"
fi
