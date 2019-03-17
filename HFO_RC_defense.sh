#!/bin/bash

~/robocin/HFO/bin/HFO --headless --defense-agents=3 --offense-npcs=3 --defense-npcs=1 --offense-team=helios --trials $1 &
sleep 10
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python torch_test_DDDQN.py &
sleep 5
python torch_test_DDDQN.py &> agent2.txt &
sleep 5
python torch_test_DDDQN.py &> agent3.txt &
sleep 5

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait