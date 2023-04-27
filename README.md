# P6-VAD
## Building and using an environment for development with Conda
### Building the environment
Create an environment:
```
conda create --name p6-vad python=3.9
```
Enter the environment:
```
conda activate p6-vad
```
Install the dependencies specified in the requirements.txt file:
```
pip install -r [PATH]\requirements.txt
```
### Developing in the environment
Enter the environment:
```
conda activate p6-vad
```
Open VSCode:
```
code [PATH].
```
Open project.

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
