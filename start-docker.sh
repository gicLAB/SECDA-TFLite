set -e

# USER_ID=$(id -u):$(id -g)
# USER=$(id -u)
# DEV_IMAGE_NAME="secda_tflite/secda_tflite:latest-dev-$USER"
# docker run --rm --privileged -it \
#     --user=$USER \
#     -v ${PWD}:/working_dir -w /working_dir \
#     $DEV_IMAGE_NAME \
#     /bin/bash


DEV_IMAGE_NAME="secda_tflite/secda_tflite:latest-dev"
docker run -it -d --name secda-tflite-dev2 -v ${PWD}:/working_dir -w /working_dir $DEV_IMAGE_NAME
# docker exec -it secda-tflite-dev2 /bin/bash -c "cd /working_dir/tensorflow/ && { echo "/root/miniconda3/bin/python"; echo ""; echo ""; echo ""; echo ""; echo ""; } | ./configure "
