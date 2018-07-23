#!/bin/bash
infile=$1
awk -f pre_process.awk "$infile" >> $infile".csv" 

