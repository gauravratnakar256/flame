# Getting started with Flame on kubernetes 


Install Kubernetes on cloudlab using following repo https://github.com/gauravratnakar256/UNIVERSE/tree/next



# Setup flame on master node

1] Install prerequisite required for running flame 

```
Run ./100-flame-setup.sh
```


2] Add flame api server url to /etc/hosts 

Example snippet:
```
<master node ip address> flame-apiserver.flame.test
```


3] Add all flame urls to coredns configmap so that pods can resolve them. Execute `kubectl edit configmap coredns -n kube-system` and add urls under loadbalance

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



4] Build flame container image

```
cd fiab
sudo ./build-image.sh
```

To check the flame image built, run `docker images`. An output is similar to:

```
REPOSITORY        TAG       IMAGE ID       CREATED        SIZE
flame             latest    f849910cb3f7   16 hours ago   4.27GB
```



5] Tag flame container image and upload it to docker registry. Before doing this login to docker repository.

```
sudo docker tag flame gaurav256/flame:p2p
sudo docker push gaurav256/flame:p2p
```



6] Update nginx ingress port and docker image details in values.yml



7] Start flame

```
./flame.sh start
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


