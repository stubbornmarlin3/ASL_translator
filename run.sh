#!/bin/bash

while true; do
    rm cookies.txt
    $HOME/test/bin/python dataset.py $?
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        break
    else
        sleep 5
    fi
done