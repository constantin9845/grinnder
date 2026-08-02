#!/bin/bash
# Exit immediately if any command fails
set -e

# 1. Clear out and reset the virtual hardware namespace
sudo nvme format /dev/nvme0 --namespace-id=1 --force
echo "Drive format done."

# 2. Format the disk with our optimized 16KB-aligned settings
sudo mkfs.ext4 -F -b 4096 -i 7000000 -O ^has_journal -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 /dev/nvme0n1

# 3. Clean up the old mount point directory and recreate it fresh
if [ -d "/mnt/nvme" ]; then
    sudo rm -rf /mnt/nvme
fi
sudo mkdir -p /mnt/nvme

# 4. Mount the drive with performance flags (Fixed trailing comma)
sudo mount -o discard,noatime,nodiratime /dev/nvme0n1 /mnt/nvme

# 5. NOW apply user permissions so your Python user can write to it
sudo chown -R femu:femu /mnt/nvme

# 6. Initialize your system environment tunings for FlexGen
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,garbage_collection_threshold:0.6,max_split_size_mb:32"
ulimit -n 65535

# 7. Execute your specific vendor-defined FDP configuration pass
sudo nvme admin-passthru /dev/nvme0n1 --opcode=239 -cdw10=6

echo "--> NVMe Drive successfully mounted and optimized at /mnt/nvme!"
