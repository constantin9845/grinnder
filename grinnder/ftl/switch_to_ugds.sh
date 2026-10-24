cd ~/uGDS/drv

sudo insmod ugds_drv.ko max_num_ctrls=64
echo 0000:00:07.0 | sudo tee /sys/bus/pci/drivers/nvme/unbind
echo 0000:00:07.0 | sudo tee /sys/bus/pci/drivers/ugds_drv/bind

echo "\n\nCheck if kernel driver set:\n"
lspci -nnk -s 00:07.0

echo "\n\nsysfs binding:\n"
ls -l /sys/bus/pci/drivers/ugds_drv/

sudo chmod 666 /dev/ugds_drv0

echo "\n\nchar device created:\n"
ls -l /dev/ugds_drv0
