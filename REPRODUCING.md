These instructions are base on those received from Code Ocean.

# Prerequisites

- Docker Community Edition (CE) ([Ubuntu install instructions](https://www.docker.com/community-edition))

# Instructions

## The computational environment (Docker image)


In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag aa303345-6d6e-40ed-928b-ccd07e111c4c; cd ..
```

> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

## Running the capsule to reproduce the results

From the main repository directory, execute the following command:
```shell
docker run --rm \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  aa303345-6d6e-40ed-928b-ccd07e111c4c ./run
```
