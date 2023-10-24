#/bin/bash

INSTANCE="affoltern"

python scripts/run_real_data.py -d ../street_network_data/${INSTANCE} -o outputs/${INSTANCE} -a betweenness_topdown
python scripts/run_real_data.py -d ../street_network_data/${INSTANCE} -o outputs/${INSTANCE} -a betweenness_biketime
python scripts/run_real_data.py -d ../street_network_data/${INSTANCE} -o outputs/${INSTANCE} -a betweenness_cartime

python scripts/run_real_data.py -d ../street_network_data/${INSTANCE} -o outputs/${INSTANCE} -a optimize