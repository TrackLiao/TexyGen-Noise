# TexyGen-Noise
Noise influence to SeqGAN

# Instruction

## Use your own dataset

put your dataset under `./data/noise/`
and rename your dataset as `data.txt`
then run the following code
```
python3 preprocess.py
```

## start training

```
bash run.sh <argument>

```

## File location

* the generated file will be in `./save/test_file.txt`
* the log will be in `./experiment.csv`
* the test result for **image_coco** data set is in `./log/image_coco_log`
* the test result for **amazon phone review** dataset is in `./log/amazone_log`

## Reference
### This project is based on https://github.com/geek-ai/Texygen


