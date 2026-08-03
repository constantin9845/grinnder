#!/bin/bash
set -e

# Format drive
sudo nvme format /dev/nvme0 --namespace-id=1 --force
echo "Drive format done."

sudo mkfs.ext4 -F -b 4096 -i 7000000 -O ^has_journal -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 /dev/nvme0n1

# Clean up the old mount point directory and recreate
if [ -d "/mnt/nvme" ]; then
    sudo rm -rf /mnt/nvme
fi
sudo mkdir -p /mnt/nvme

# Mount the drive 
sudo mount -o discard,noatime,nodiratime /dev/nvme0n1 /mnt/nvme

# user permissions 
sudo chown -R femu:femu /mnt/nvme

# number of files
ulimit -n 65535

echo "--> NVMe Drive mounted at /mnt/nvme"
