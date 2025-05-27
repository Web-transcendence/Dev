#!/bin/sh

cd /app/core/Client
./myFactory &

cd /app/core/AiServer
./myDQN &

wait -n

exit $?