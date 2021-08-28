# Kaggle BMS Molecular Translation (kyamaro: KF part)

## Preparation

### Hardware

- OS: Ubuntu 18.04.5
- GPU: NVIDIA A100
- NVIDIA Driver: 455.45.01
- CUDA: 11.1
- Docker: 19.03.13
- docker-compose: 1.22.0

### Software

Required softwares are included in the Dockerfile.
You can build Docker image as follows:

```
docker-compose build
```

### Data

Set the following environment variables in .env

```
KAGGLE_USERNAME=xxxxx
KAGGLE_KEY=xxxxx
```

Then you can download datasets as follows:

```
docker-compose run --rm cpu exec/download.py
```

## Training

```
./run/train.sh
```

## Candidate generation

```
./run/candidate_generation.sh
```

## Candidate rescoring

```
./run/candidate_rescoring.sh
```
