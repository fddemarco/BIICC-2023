#!/bin/bash

args="-t submissions -wd /media/franco/TOSHIBA/pushshift/data"
for year in {2012..2018}
do    
    python ../src/experiments/app.py "$args" -y "$year" -e embeddings -o results
    python ../src/experiments/app.py "$args" -y "$year" -e compare -o results
done

