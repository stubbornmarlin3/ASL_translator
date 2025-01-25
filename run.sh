#!/bin/bash

index=0

while true; do
    rm cookies.txt
    index=$($HOME/test/bin/python dataset.py $index | tee >/dev/tty | tail -n 1)
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        break
    else
        sleep 5
    fi
done