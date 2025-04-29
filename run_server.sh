#!/bin/bash

# Activate virtual environment
source env/bin/activate


# Start a new screen session named 'myapp' and execute the application serving command within it
screen -S myapp -d -m bash -c 'waitress-serve --port=5000 --call app:create_app > myapp.log 2>&1'


