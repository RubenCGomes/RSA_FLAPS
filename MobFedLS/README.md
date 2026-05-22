# MobFedLS

Federated Learning framework for smart cities scenarios

## How to clone the repo

The flag `--recurse-submodules` is used to get the flower folder/submodule, only needed to build the mfls-server and mfls-ghostclient modules

```bash
git clone --recurse-submodules https://code.nap.av.it.pt/ai4sme/MobFedLS.git
```

## Introduction and Description

The MobFedLS architecture consists of three components: the MFLS–Interface, the MFLS–Manager, and the MFLS–Maker, that is composed by the MFLS–Server and the MFLS–GhostClient. Components and subcomponents of MobFedLS were implemented in Docker containers as microservices, which allows them to be easily used on devices ranging from the edge to the cloud.

![MobFedLS System Overview](https://i.ibb.co/pK7YP39/Demo-Architecture-Simple-Architecture.png)

- The MFLS–Interface is a bridge between the FL and the ML model, allowing greater independence between FL and ML activities, using a loose coupling approach between ML and FL. At this moment each ML application that wants to use a Federated approach needs a new MobFedLS instance.

- The MFLS–Manager is the orchestrator that coordinates the search for neighbours (mobile and fixed), manages the selection of clients for the federated process, and activates the instantiation process of federated components on the selected clients. MFL–Manager addresses the mobility challenge by dynamically finding and preparing mobile nodes on demand. Furthermore, at the end of the federated process, MFL–Manager proceeds to eliminate the FL instances on all participating nodes.

- The MFLS–Server is the central hub for the FL process itself. It works as an aggregator. During the federated process, only one instance of the MFLS–Server will be active. This instance will be created on the node that activates the federated process, with this node being seen as the master node.

- The MFLS–GhostClient consists of a client service named Ghost due to the absence of the ML model within it. The MFLS–GhostClient works as an intermediary only during the FL rounds between the client’s ML models and the MFLS–Server that will aggregate their parameters, facilitating the connection between the MFLS–Server and the MFLS–Interface during the rounds. This implies that, when the MFLS–Server initiates a request to the clients for model fitting or evaluation, the MFLS–GhostClient does not directly perform the action on the model.

The MFLS-Server and MFLS-GhostClient are built with the [Flower Framework](https://flower.ai).

### Already created and coded ML-Apps

List of ML-Apps already created:

- [ATCLL Hourly Forecast of number of cars entering the city of Aveiro](ml-apps/clientsML_forecast/README.md)
- [MNIST Database](ml-apps/clientsML_mnist/README.md)

For future and under development features and ML-Apps see [Project Status and Roadmap](#project-status-and-roadmap)

## Images to pull

The main modules of the framework have a pre-built image ready to use. To get them, run:

```bash
docker pull code.nap.av.it.pt:5050/ai4sme/mobfedlearnsys/find-neighbours
docker pull code.nap.av.it.pt:5050/ai4sme/mobfedlearnsys/mfls-manager
docker pull code.nap.av.it.pt:5050/ai4sme/mobfedlearnsys/flower-server
docker pull code.nap.av.it.pt:5050/ai4sme/mobfedlearnsys/flower-ghostclient
```

## Usage

Steps for use MobFedLS:

1. [Code the ML model and necessary functions](#1-build-the-ml-model-script)
2. [Build ML-App Docker Image](#2-build-ml-app-image)
3. [Create and fill the .env file](#3-create-env-file)
4. [Run `docker compose up` -d command](#4-information-on-docker-composeyaml-file)
5. [Triggering Federation](#5-triggering-federation-mnist-example)
6. [Check the logs](#6-logs)

To try out an already built ML-App, select one and begin from step 2. Every time a change is made, run `docker compose down`, delete de image of the ML-App, using `docker rmi ML-App`, and restart from step 2.

### 1. Build the ML model script

Create a folder inside the ml-apps folder, with the name of the app. Inside that folder should exist (if the app will be containerized):

- datasets (folder)
  - all dataset files and folders necessary (can be .csv, .json, .yaml with the path for the dataset)
- `clientML.py` (file)
- `Dockerfile.clientML_AND_interface` (file, see [MNIST Example - Simple](ml-apps/clientsML_mnist/Dockerfile.clientML_AND_interface))
- `requirements_clientML.txt` (file)

Other files can be necessary, such as, a sh script for preload some variables for the Docker container, a README.md, etc.

Using any library specify the structure of the model wanted. Then there are some obligatory functions with the following structure (explained using tensorflow library):

```python
model = tensorflow.keras.models.MODEL
x_train, y_train = [] # Train Partition
x_val, y_val = [] # Validation Partition
x_test, y_test = [] # Test Partition
x_pred, y_pred = [] # Predict Partition

def get_parameters():
    global model
    parameters = model.get_weights()
    return parameters

def set_parameters(parameters):
    global model
    model.set_weights(parameters)

def get_data():
    # Separate the data into Training, Validation and Testing (last one is optional)
    # Populate the x_train, y_train; x_val, y_val; x_test, y_test; x_pred, y_pred as needed

def fit(config): # config example:  {'batch_size': '32', 'epochs': '30', 'epoch_global': '2'}
    return len(x_train), results # number of examples on the training partition

def evaluate(config): # config example
    return float(loss), len(x_test), {"metric_1": float(metric_1)}

def predict(plot_graphs, before_after):
    return {"mse": float(mse), "mae": float(mae)}, {"r2": float(r2)}

# Mandatory, and just live it like this
def in_usage(value):
    global occupied
    occupied = value
    if value == 1:
        log.info("ML app is now in use, and blocked")
    elif value == 0:
        log.info("ML app is now freed")

def set_run_path(run_path):
    global current_run_path
    current_run_path = run_path
    log.info("Current run path set")
```

All of the functions are built to work with the MFLS-Interface. For more details see the simple example of - [MNIST Database](ml-apps/clientsML_mnist/clientML.py) and the [MFLS-Interface Specifications](interface/interface.py). The fit and evaluate `config` and return parameters are complaint with the [flower client architecture](https://flower.ai/docs) ([MFLS-GhostClient](ghostclient/client-flower.py)).

### 2. Build ML-App Image

The only additional image that is required to build is the ml-app image. Run the following command changing the < EXAMPLE > placeholder accordingly:

```bash
docker build -t ml-app-<EXAMPLE> -f ml-apps/clientsML_<EXAMPLE>/Dockerfile.clientML_AND_interface .
```

(See [Build from source section](#build-from-source) for more build commands.)

#### 2.1 Build MNIST Example (For example)

To clone and build the MNIST Example do:
```bash
cd MobFedLS/tools/ml-apps
git clone https://code.nap.av.it.pt/ai4sme/mobfedlearnsys/ml-apps/mnist.git clientsML_mnist
cd clientsML_mnist
docker build -t ml-app-mnist -f deployment/Dockerfile .
```

### 3. Create .env file

Once steps 1 and 2 were repeated on all machines its time to start configuring each one accordingly.
Create a .env file which is required to pass some variables to inside the containers, as it is possible to see in the docker-compose file. The obligatory variables are:

```bash
MACHINE_ID=... >>> update per machine
USERNAME=...
PASSWORD=...
LOGGING_LEVEL=TMSTMP # (it can also be: DEBUG; INFO; etc.)
PLOT_PERFORMANCE=False
REGISTRY_URL=code.nap.av.it.pt:5050/ai4sme/mobfedlearnsys/
SERVER_IMG=flower-server
CLIENT_IMG=flower-ghostclient
```

For the ML application container, is necessary to add the following variables:

```bash
ML_APP_IMG=ml-app...
ML_BASE_DIR=clientsML...
DATASET=... >>> for the mnist example use the configX.csv
```

Other variables can be used for ease of deployment. Don't forget to add the new variables also on the docker-compose file.

### 4. Information on docker-compose.yaml file

On the provided docker-compose.yaml file, each container has the necessary environment variables and volumes for the pre-built containers.

For the ML-App the most common environment variables are:

```dockerfile
environment:
    - LOGGING_LEVEL
    - DATASET
    - PLOT_PERFORMANCE
```

Add more as needed. Also if the host machine has a gpu and is possible to be used, check how to enable the gpu passthrough to inside the container. For example, for the Nvidia Jetson XXX use the the flags on the `docker-compose.yaml` file on the ML-App container:

```dockerfile
ipc: host
runtime: nvidia
````

Also for the ML-App, the following volumes are needed, for dataset upload, docker usage and logs:

```dockerfile
volumes:
    - './ml-apps/$ML_BASE_DIR/datasets:/app/dataset'
    - '/var/run/docker.sock:/var/run/docker.sock'
    - './logs:/app/logs'
````

To run the containers in the background:

```bash
cd ~/MobFedLS/deployment
docker compose up -d
```

To check if containers were launched correctly:

```bash
docker compose logs
```

To stop:

```bash
docker compose down
```

### 5. Triggering Federation (Mnist Example)
#### 5.1 Starting the local model

After completing step 3 and 4 it's now possible to interact with this instance of the MobFedLS framework.

Each machine exposes two fastAPI docs pages where the list of available endpoints and interact with them are available.

<http://{machine-ip}:5001/docs> --> ml-app API
<http://{machine-ip}:5101/docs> --> mfls-manager API

To first try on the tool lets ensure we already have a running ML model that is fitted and it has context of the data produced by a “sensor”.

In order to do this, it's needed to initialize the parameters according to the dataset of each client. Use the endpoint `/internal/get_data` and then the endpoint `/internal/fit` of the ml-app API to do an initial training of the model. Make sure to only change the client number so it matches the X in the configX.csv

#### 5.2 Setting up the server

Choose one of the machines to be the server and run the following commands.

```bash
docker compose down
cd ~/MobFedLS/cmd/findneighbours/cmd/findNeighbours/neighbours_lists/
vim test1.json
```

test1.json:

```json
{
    "0": "<server-node-ip>",
    "1": "<server-node-ip>",
    "2": "<machine2-ip>",
    "3": "<machine3-ip>",
    "4": "<machine4-ip>"
}
```

Then restart the node with:

```bash
docker compose up -d
```

Repeat steps 5.1

#### 5.3 Watching the Federation Process take place

On each machine run:

```bash
sudo watch -n 1 docker ps
```

It's possible to see new containers appear on each machine as the FL process takes place.

Head to `/external/start_aggregation` (ml-app API) on the server node and fill the fields as follows:

```json
n_rounds: 5
round_timeout: 100.0
fl_algorithm: FedAvg
n_epochs: 32
batch_size: 32
neighbours_file: test1.json
```

And then press the execute button.

### 6. Logs

The log files are saved on the folder `assets/logs`. They are organized by date, so check the todays date and the different runs are inside a numbered folder begining with the first run of the day on folder `run0`.
Example: `logs/2024-07-12/run0`

---

## Project Status and Roadmap

The MobFedLS project is still under active development and is frequently updated to introduce new features and correct issues. Please report any problems encountered.

Features under development or on hold:

- After each round ensure that the parameters/weights apllied to the model are the best ones, that is, before or after the FL Rounds/Process (Working on)

ML-Apps under development or on hold:

- ATCLL Camera Object Detection Service YOLO ([containerized](ml-apps/clientsML_yolo-jetson/README.md) | [terminal](ml-apps/clientsML_yolo-terminal/README.md)) (Working on)
- [Traffic Sign Detection](ml-apps/clientsML_traffic/README.md) (Incomplete - Currently stopped)

## Authors and Support

Questions and Bug Reports: @Bernardo Barreto / <bernardo.barreto@av.it.pt>

## Extras

### Build from source

If build from scratch is needed, use the following docker commands (dont forget the [ML-App Image](#2-build-ml-app-image)):

- Find-Neighbours

```bash
docker build -t find-neighbours -f findNeighbours/Dockerfile.find-neighbours .
```

- MFLS-Manager

```bash
docker build -t mfls-manager -f manager/Dockerfile.manager .
```

- MFLS-Server

```bash
docker build -t mfls-server -f server/Dockerfile.server-flower .
```

- MFLS-GhostClient

```bash
docker build -t mfls-ghostclient -f ghostclient/Dockerfile.client-flower .
```

---
