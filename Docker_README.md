 sudo -E docker build --progress=plain -t my_image .

sudo docker run --gpus all -it --entrypoint bash my_image

follow that for installing the nvidia docker toolkit on your host so u will able to share the gpus to the docker
https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab


for downlaoding the weights and checkpoints run the docker first with the following flags
sudo docker run -e DOWNLOAD_MODELS=True -e MODELS_URL="https://mega.nz/file/AiFzBSDS#BqcKazpnYaS0GR4i2HqHCsenbowzr9KjeQQ9X2VPFHY" -it my_image

for running the docker and obening a simple bash
sudo docker run --gpus all -it --entrypoint bash my_image

the --gpus all is for the docker will uses the gpus
