#!/bin/bash

# Install Helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh 

# Install jq
sudo apt update -y
sudo apt-get install -y jq

# Install Ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.1.3/deploy/static/provider/baremetal/deploy.yaml

# Install Certificate Manager
./setup-cert-manager.sh

# install moreutils (sponge)
sudo apt-get install -y moreutils 
