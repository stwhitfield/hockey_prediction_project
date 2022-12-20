#!/bin/bash

echo "TODO: fill in the docker run command"
docker container run -e COMET_API_KEY=$COMET_API_KEY -p 30001:30001 ift6758/serving:1.0.0