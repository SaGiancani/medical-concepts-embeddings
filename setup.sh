#!/bin/bash
sudo apt update
sudo apt install python3.8 & sudo apt install python3-pip
wait
pip install virtualenv
wait 
virtualenv venv
sleep 2
source venv/bin/activate
sleep 2
cat requirements.txt | xargs -n 1 pip install
wait