

L=$1
R=$2
TP=$3
BZ=$4

model_path=""
data_path=""
output_dir=""


python STILL-3-TOOL/data_synthesis/coding_by_thinking.py \
        --l_split $L \
        --r_split $R --tp $TP --bz $BZ \
        --model_path $model_path \
        --data_path $data_path \
        --output_dir $output_dir