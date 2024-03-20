#!/bin/bash

echo $date
echo $hostname

# pass the rest of the args
cd /home/hltcoe/oweller/my_exps/mteb-instruct
all_but_first_input="${@:1}"
echo "All but first input: $all_but_first_input"
python=~/.conda/envs/mteb-instruct/bin/python
cmd=$($python evaluate_any.py $all_but_first_input)

# replace `python` in cmd with `$python`
cmd=${cmd/python/$python}
echo "Running command: $cmd"

# run that command
$cmd
echo $date