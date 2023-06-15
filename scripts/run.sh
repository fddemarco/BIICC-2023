#!/bin/bash

# Set the common arguments
args="-t submissions -wd /media/franco/TOSHIBA/pushshift/data"


#    input_dir="/media/franco/TOSHIBA/pushshift/data/pushshift-reddit/2012/results/subreddits.txt"
#    output_dir="/media/franco/TOSHIBA/pushshift/data/pushshift-reddit/2012/results/subreddits"
#    fasttext skipgram -input $results_dir -output subreddits -epoch 1 -dim 300

# Loop over the years and execute the script
for year in {2012..2012}
do    
    python app.py $args -y $year -e embeddings -o results
    python app.py $args -y $year -e compare -o results
done

