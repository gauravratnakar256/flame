# Getting started with Flame on Kubernetes 


Install Kubernetes on cloudlab using following repo https://github.com/gauravratnakar256/UNIVERSE/tree/next


# Setup Flame on master node

### 1] Clone the repository
```
git clone https://github.com/gauravratnakar256/flame.git && cd flame/fiab
```

### 2] Install prerequisite required for running flame 

```
./500-flame-setup.sh
```


### 3] Add flame api server url to /etc/hosts 

Example snippet:
```
sudo vi /etc/hosts
<master node ip address> flame-apiserver.flame.test
```


### 4] Add all flame urls to coredns configmap so that pods can resolve them. 

Execute `kubectl edit configmap coredns -n kube-system` and add urls under loadbalance

Example snippet:
```
hosts {
      <master node ip address>  flame-apiserver.flame.test
      <master node ip address>  flame-notifier.flame.test
      <master node ip address>  flame-mlflow.flame.test
      <master node ip address>  flame-controller.flame.test
      <master node ip address>  minio.flame.test
      fallthrough
    }
```

For complete core dns config map example [refer this](https://github.com/gauravratnakar256/flame/blob/main/images/coredns_configmap.png)

### 5] Build flame container image

```
sudo ./build-image.sh
```

To check the flame image built, run `sudo docker images`. An output is similar to:

```
REPOSITORY        TAG       IMAGE ID       CREATED        SIZE
flame             latest    f849910cb3f7   16 hours ago   4.27GB
```


### 6] Tag flame container image and upload it to docker registry.

Before doing this login to docker repository.

```
sudo docker tag flame gaurav256/flame:p2p
sudo docker push gaurav256/flame:p2p
```



### 7] Update nginx ingress port and docker image details in values.yml

Execute `kubectl get svc ingress-nginx-controller -n ingress-nginx` to get ingress port associated with 80 and 443

Example snippet:
```
NAME                       TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)                      AGE
ingress-nginx-controller   NodePort   10.107.175.66   <none>        80:32685/TCP,443:31697/TCP   27d
```

Update values in control/values.yaml

```
vi helm-chart/control/values.yaml
```
Update nginxhttps port with port associated with 443 and nginxhttp with port associated with 80

Example snippet:
```
endpointports:
  http: 30205
  https: 30371
  nginxhttps: 31697
  nginxhttp: 32685
```

Update imageName, imageTag, workerImageName and workerImageTag with docker image details created in step 6.

Example snippet:
```
imageName: gaurav256/flame
imageTag: p2p
workerImageName: gaurav256/flame
workerImageTag: p2p
```

Update values in deployer/values.yaml

```
vi helm-chart/deployer/values.yaml
```

Update imageName, imageTag, nginxhttps and nginxhttp values same as control/values.yaml.


### 8] Start flame

```
sudo ./flame.sh start
```

Check that all pods were created successfull `kubectl get pods -n flame`

Example output:

```
NAME                                      READY   STATUS    RESTARTS      AGE
flame-apiserver-58b755c5db-jnvh2          1/1     Running   0             11h
flame-controller-665749845d-hvrh5         1/1     Running   2 (11h ago)   11h
flame-deployer-default-77555cc984-82f7j   1/1     Running   0             11h
flame-metaserver-848b876b8-kct2j          1/1     Running   0             11h
flame-minio-565f9445f8-62z8w              1/1     Running   0             11h
flame-mlflow-57485f7d55-ddt44             1/1     Running   0             11h
flame-mongodb-0                           1/1     Running   0             11h
flame-mongodb-1                           1/1     Running   0             11h
flame-mongodb-arbiter-0                   1/1     Running   0             11h
flame-mosquitto-668b55d77f-2bjzj          1/1     Running   0             11h
flame-mosquitto2-5d4f64bdbc-dqcxp         1/1     Running   0             11h
flame-notifier-8f8f5d855-9q6p2            1/1     Running   0             11h
postgres-76d7b48888-qcdv9                 1/1     Running   0             11h
```

# Setup flamectl on master node

### 1] Execute build-config.sh
 
```
./build-config.sh <flame api server url>:<nginxhttps port>
```
Example snippet:
```
./build-config.sh flame-apiserver.flame.test:31697
```

### 2] Install go or update to version 1.19.1
```
wget https://go.dev/dl/go1.19.1.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.19.1.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
```

### 3] Compile flamectl package
```
cd .. 
make install
```

### 4] Add flamectl path to $PATH
```
export PATH="$HOME/.flame/bin:$PATH"
```

# Run MedMNIST example

To manually run medmnist job refer to instructions at [medmnist example](https://github.com/gauravratnakar256/flame/blob/main/examples/medmnist/README.md).

To run medmnist job using script  follow below command

```
cd  examples/medmnist
./502-run-medmnist.sh <code zip name> <number of trainers>
```

For full data code zip name is `medmnist` and for dummy data use `medmnist_dummy`
After running the script you will get jobId use it for below commands

Example output:
```
New job created successfully
        ID: 6335234925b853b07d3511df
        state: ready
```

Use below command to start the job
```
flamectl start job <jobId>  --insecure
```

To check status of job
```
flamectl get jobs --insecure
```