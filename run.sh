#!/bin/bash

exit_code=0

while true; do
    rm cookies.txt
    $HOME/test/bin/python dataset.py $exit_code
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        break
    else
        sleep 5
    fi
done