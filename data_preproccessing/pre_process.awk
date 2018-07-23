#!/bin/awk -f
BEGIN {
FS=",";
}  
{
print $4 $5 $6 $7 $8;
}       
END{} 

 
