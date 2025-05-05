#!/bin/bash

KEY=$(openssl rand -hex 64)
echo "SECRET_KEY=$KEY" > .env