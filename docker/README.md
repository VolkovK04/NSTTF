
# Build docker image for testing environment (RUN ONCE)

```bash
docker build -t nsttf/nsttf:1.0 -f Dockerfile .
```

Launch testing environment. (make sure the docker daemon is running)

```bash
docker run -it --rm --name=nsttf --mount type=bind,source="${PWD}/..",target=/src nsttf/nsttf:1.0 bash 
```

Run test script
```bash
cd src/docker
python3 run_tests.py <-san>
```