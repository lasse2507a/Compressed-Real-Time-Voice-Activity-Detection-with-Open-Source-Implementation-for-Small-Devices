# P6-VAD
##

## Docker
Build the Docker image locally using the following command:
```
docker build -t p6-vad:1.x.x(version) .
```
Run the Docker container using the following command:
```
docker run --name p6-vad-container -d -v ${pwd}:/code p6-vad-image sleep infinity
```
In VSCODE using the extension Dev Containers attach the running container to start a dev environment.
