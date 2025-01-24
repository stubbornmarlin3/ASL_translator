#!/bin/bash

index=0

while true; do
    rm cookies.txt
    index=$($HOME/test/bin/python dataset.py $index | tee >(cat) / tail -n 1)
    
    if [ $index -eq 0 ]; then
        break
    else
        sleep 5
    fi
done