#/bin/bash

export MY_CONTAINER="MLU270-tianyi"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
#xhost +
if [ 0 -eq $num ];then
    docker run -e DISPLAY=unix$DISPLAY --device /dev/cambricon_dev0 --pid=host \
               --net host \
               -v /sys/kernel/debug:/sys/kernel/debug \
               -v /tmp/.X11-unix:/tmp/.X11-unix -it --privileged=true --name $MY_CONTAINER \
               -v /home/scty/work_space/cambricon:/opt/cambricon \
                  yellow.hub.cambricon.com/pytorch/pytorch:0.11.114-ubuntu18.04 /bin/bash
else
    docker start $MY_CONTAINER
    #sudo docker attach $MY_CONTAINER
    docker exec -ti $MY_CONTAINER /bin/bash
fi

