# P6-VAD
## Docker
You can build the Docker image using the following command:
'''
docker build -t p6-vad-image .
'''
And then run the Docker container using the following command:
'''
docker run --name p6-vad-container -d -v ${pwd}:/code p6-vad-image sleep infinity
'''