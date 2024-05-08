# Docker setup

This docker setup is tested on Ubuntu.

make sure you are under directory yourworkspace/instantmesh/

Build docker image:

```bash
docker build -t instantmesh -f docker/Dockerfile .
docker build -t instantmesh_curl -f docker/DockerfileCurl .
```

Run docker image with a local model cache (so it is fast when container is started next time):

```bash
mkdir -p $HOME/models/
export MODEL_DIR=$HOME/models/

docker run -it -p 43839:43839 --platform=linux/amd64 --gpus all -v $MODEL_DIR:/workspace/instantmesh/models instantmesh
```

To use specific GPUs:

```bash
mkdir -p $HOME/models/
export MODEL_DIR=$HOME/models/

docker run -it -p 43839:43839 --platform=linux/amd64 --gpus '"device=0,1"' -v $MODEL_DIR:/workspace/instantmesh/models instantmesh
```

Navigate to `http://localhost:43839` to use the demo.

To run a curl API:

```bash
mkdir -p $HOME/models/
export MODEL_DIR=$HOME/models/

docker run -it -p 8112:8112 --platform=linux/amd64 --gpus '"device=0,1"' -v $MODEL_DIR:/workspace/instantmesh/models -e AUTH_TOKEN=your_auth_token -e PORT=8112 instantmesh_curl
```
