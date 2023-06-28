#!/bin/bash

# Set the common arguments

# Loop over the years and execute the script

path = "/media/franco/TOSHIBA/pushshift/data/pushshift-reddit"

for year in {2013..2018}
do    
    mkdir $path/$year/truncated/results
    args="-input /media/franco/TOSHIBA/pushshift/data/pushshift-reddit/2013/truncated/subreddits.txt -output /media/franco/TOSHIBA/pushshift/data/pushshift-reddit/2013/truncated/results/subreddits -pretrainedVectors /media/franco/TOSHIBA/pushshift/pretrained/wiki.en.vec -dim 300"
    ~/Downloads/fastText-0.9.2/fasttext skipgram $args
done



