#!/bin/bash

# Keeping the model file name passed as argument while running this bash script.
model_file_name=$1

# Keeping the timestamp at which program was run so that all its output files remain in one folder named by the timestamp
program_time_stamp=$2

echo $program_time_stamp

# Run BioNetGen simulation.
echo "Running BioNetGen simulation..."
./simulators/BioNetGen-2.3/BNG2.pl --outdir output/$program_time_stamp/ ./models/$program_time_stamp/$model_file_name.bngl

# Read the contents of the gdat file.
gdatContents=`cat output/$program_time_stamp/*$model_file_name*.gdat`

# Echo contents as test
#echo "Contents of gdat:\n"
#echo ${gdatContents}

messagePath_temp=output/$program_time_stamp/trace.txt
messagePath_final=output/$program_time_stamp/trace.csv
# Write the message to a temporary file.
echo -n "${gdatContents}" > ${messagePath_temp}

# Generate the final csv file compatible with TeLEX (deleting # from first row, squeezing all spaces in file to one space, replacing that one
# space by comma, removing the first comma from every line)
cat ${messagePath_temp} | tr -d '#' | tr -s ' ' | tr ' ' ',' | sed 's/^,//' > ${messagePath_final}

