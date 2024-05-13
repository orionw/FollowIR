#!/bin/bash

cd /home/hltcoe/oweller/my_exps/FollowIR

echo $date
echo $hostname

# pass the rest of the args
all_but_first_input="${@:1}"
echo "All but first input: $all_but_first_input"
cmd=$(~/.conda/envs/followir/bin/python evaluate_any.py $all_but_first_input)

# replace `python` in cmd with `$python`
# cmd=${cmd/python/$python}
echo "Running command: $cmd"

# run that command
$cmd
echo $date