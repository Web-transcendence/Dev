#!/bin/bash

mkdir -p data/user-management
mkdir data/social
mkdir data/pong
mkdir data/towerDefense

KEY=$(openssl rand -hex 64)
echo "SECRET_KEY=$KEY" > .env