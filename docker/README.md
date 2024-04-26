
# Build docker image for testing environment

```bash
docker build -t nsttf/nsttf:1.0 -f Dockerfile .
```

Launch testing environment.

```bash
docker run -it --rm --name=nsttf --mount type=bind,source="${PWD}/..",target=/src nsttf/nsttf:1.0 bash 
docker run -it --rm --name=nsttf -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source="${PWD}/..",target=/src nsttf/nsttf:1.0 bash 
```
