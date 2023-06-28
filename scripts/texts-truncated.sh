#!/bin/bash

args="-t submissions -wd /media/franco/TOSHIBA/pushshift/data -e texts -o truncated"
for year in {2015..2018}
do    
    python ../src/experiments/app.py $args -y $year -e truncate
done

