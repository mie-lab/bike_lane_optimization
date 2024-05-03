# build stage
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /backend

# Install GDAL dependencies
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools
RUN pip install --no-binary fiona -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port (if needed)
EXPOSE 8989

# Define the command to run the Flask application
CMD ["python", "app.py"]
