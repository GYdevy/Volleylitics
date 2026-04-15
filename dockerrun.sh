docker run -it \
          --device=/dev/kfd \
          --device=/dev/dri \
          --group-add video \
          --ipc=host \
          --shm-size=8G \
          -v ~/projects/Volleylitics:/workspace \
          -v /mnt/hdd/videos:/videos \
          -v /mnt/hdd/datasets:/datasets \
          volleylitics
