#!/bin/bash

total_iterations=20
source activate parameter-synth

if [ -n "$n" ]; then
  total_iterations=$n
else
  total_iterations=20
fi

for ((i=1; i<=$total_iterations; i++))
do
    python param-synth.py $spec $i &
done
