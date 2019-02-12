#!/bin/bash
# python main.py -g seqgan -t real -d ./data/noise/image_coco.txt -f ./data/noise/swap.txt
python main.py -g seqgan -t real -d ./data/noise/image_coco.txt -f ./data/noise/reverse.txt
# python main.py -g seqgan -t real -d ./data/noise/image_coco.txt -f ./data/noise/misword.txt
# python main.py -g seqgan -t real -d ./data/noise/image_coco.txt -f ./data/noise/wtense.txt
# python main.py -g seqgan -t real -d ./data/noise/image_coco.txt -f ./data/noise/typo.txt
