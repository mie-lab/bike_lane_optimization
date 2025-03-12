#!/bin/bash

# Define the code snippet to be added

# code_to_append='
# def create_app():
#     return app
# '

# Append the code to the end of the prod/app.py file
echo "$code_to_append" >> app.py


# Activate virtual environment
source env/bin/activate


# Start a new screen session named 'myapp' and execute the application serving command within it
screen -S myapp -d -m bash -c 'waitress-serve --port=5000 --call app:create_app > myapp.log 2>&1'


