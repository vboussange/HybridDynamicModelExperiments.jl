#!/bin/bash
# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process

file=$1
arg=$2  # Optional prefix argument

namesim_with_ext=$(basename ${file})
namesim="${namesim_with_ext%.*}"

mkdir -p stdout

echo "Launching script for $namesim"
chmod +x $file

nohup julia --project=. $file $arg > "stdout/${namesim}.out" 2>&1 &
echo $! > "stdout/${namesim}_save_pid.txt"
