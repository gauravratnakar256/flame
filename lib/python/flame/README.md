# A guide to run python library locally without engaging with the system components

## Environment Setup
We recommend setting up your environment with `conda`. This example is based on Ubuntu 22.04.
```bash
conda create -n flame python=3.9
conda activate flame


pip install google
pip install tensorflow
pip install torch
pip install torchvision

cd ..
make install
```

## Run Hierarchical FL Job With MQTT backend

## Install mosquitto broker

```bash
sudo apt-get update
sudo apt-get install mosquitto
```

Update mosquitto broker URL in config.json of trainers, middle_aggregator and top_aggregator. It will be localhost with no port for local machine.

## Run Trainers

```bash
cd examples/hier_mnist/trainer

python main.py config_us.json

#Open another terminal and run

python main.py config_uk.json

```

## Run Middle Aggregator

```bash
cd examples/hier_mnist/middle_aggregator

python main.py config_us.json

#Open another terminal and run

python main.py config_uk.json

```

## Run Top Aggregator

```bash
cd examples/hier_mnist/top_aggregator

python main.py config.json
```

## Run Hierarchical FL Job With Grpc backend

Go to flame/cmd/metaserver and run  `go run main.go`. This will start metaserver which will run on port 10104. Update backend as p2p for grpc in configs of all trainers and aggregators. Start the job similar to mqtt backend


