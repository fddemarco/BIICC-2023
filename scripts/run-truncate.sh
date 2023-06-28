#!/bin/bash

args="-t submissions -wd /media/franco/TOSHIBA/pushshift/data"
for year in {2013..2018}
do    
    python ../src/experiments/app.py $args -y $year -e truncate
done

