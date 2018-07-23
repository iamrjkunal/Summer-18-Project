#!/bin/bash
infile=$1

if [ -d "$infile" ] ; then
  echo its a directory
elsif [ -f "$infile" ], then
  echo its a file
else
  echo some other type
fi
