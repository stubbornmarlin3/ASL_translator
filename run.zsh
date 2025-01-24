#!/usr/bin/env bash

while [$? -ne 0]; do
    rm cookies.txt
    sleep 5
    $HOME/test/bin/python dataset.py $?
    
done