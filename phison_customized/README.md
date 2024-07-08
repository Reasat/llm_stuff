# aiDAPTIV Install SOP and Usage
## **Requirements**

- Nvidia driver version 535 installation
- CUDA 12.2.1 installation

## **Install Driver and Cuda (Optional)**

- Nvidia GPU
    - Install Driver (Estimated time: 5 min)
        
        ```bash
        sudo apt install nvidia-utils-535
        sudo apt install nvidia-driver-535
        
        sudo reboot
        # Verify if the installation is successful.
        nvidia-smi
        
        ```
        
    - Install Cuda (Estimated time: 15min)
        
        ```bash
        cd /home/$USER && wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda_12.2.1_535.86.10_linux.run
        sudo sh ./cuda_12.2.1_535.86.10_linux.run
        # Continue > accept > Look at the images below and do not check the box for Driver as it has already been installed. Then, move the cursor to Install and press Enter.
        ```
        

## **Download setup.sh**

```bash
wget http://ec2-43-207-140-249.ap-northeast-1.compute.amazonaws.com:8767/setup.sh
```

## **Deploy aiDAPTIV and related library**

```bash
# And select 2 to deploy aiDAPTIV+
bash setup.sh
```

To make the environment variables take effect, you can use the command **source ~/.bashrc**

## Get LLM Model (Optional)

```bash
# Select 1 to download Llama Model
bash setup.sh
```

## **Test and Validation**

1. Log in Huggingface (Estimated time: 1 min)
    
    ```bash
    git config --global credential.helper store
    huggingface-cli login
    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
    
    ```
    
2. Create Raid0 (optional)
    
    ```bash
    sudo apt install mdadm
    #To create a RAID 0 array, specify the disk locations based on the desired number of disks. Note: provide the raw paths, not partitions.
    
    #example sudo mdadm --create /dev/md0 --level=0 --raid-devices=2 /dev/nvme0n1 /dev/nvme1n1
    sudo mdadm --create /dev/md0 --level=0 --raid-devices=<num_disk> </dev/disk1> </dev/disk2> ...
    
    #Format the disk.
    sudo mkfs -t ext4 /dev/md0
    #Mount the disk.
    sudo mkdir -p /media/user/nvme0
    sudo mount /dev/md0 /media/user/nvme0
    sudo chown -R $USER:$USER /media/user/nvme0
    
    ```
    
    If you need to dissolve a RAID 0 array or delete data: (optional)
    
    ```bash
    #Dissolve RAID 0
    sudo mdadm --stop </dev/disk>
    
    #Clear RAID 0 data
    sudo mdadm --zero-superblock </dev/disk>
    
    ```
    
3. Mount Disk (optional)
    
    ```bash
    #Format the disk. If prompted, enter "y" to confirm.
    sudo mkfs -t ext4 /dev/<Disk>
    #Create the target directory.
    sudo mkdir -p /media/user/nvme0
    #Mount the disk.
    sudo mount /dev/<Disk> /media/user/nvme0
    #Change the folder ownership.
    sudo chown -R $USER:$USER /media/user/nvme0
    ```
    
4. Run training
    
    ```bash
    #command example
    phisonai --num_gpus 2 \
        --model_name_or_path /home/$USER/Desktop/llm/Llama-2-7b-hf \
        --data_path Dahoas/rm-static \
        --nvme_path /media/user/nvme0 \
        --per_device_train_batch_size 2 \
        --num_train_epochs 1 \
        --output_dir /media/user/nvme0/result/
    ```
    

## aiDAPTIV Document

## **Introduction**

The PhisonAI SDK provides a custom framework for LLM GPU training, which is an optimized deep learning training library. This document outlines the usage method and the input and output parameters of the SDK.

## **How to Usage**

### **Command Example**

Here is an example of how to use the SDK:

```bash
phisonai --num_gpus <gpu_count> \
            --model_name_or_path <model_path> \
            --data_path <dataset_location> \
            --nvme_path <swap_disk_location> \
            --per_device_train_batch_size <batch_size> \
            --num_train_epochs <epochs> \
            --output_dir <output_model_path>

```

example:

```bash
phisonai --num_gpus 2 \
    --model_name_or_path /home/$USER/Desktop/llm/Llama-2-7b-hf \
    --data_path Dahoas/rm-static \
    --nvme_path /media/user/nvme0 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --output_dir /media/user/nvme0/result/
```

## **Parameters introduction**

1. Input Parameters
    - GPU Count: The number of GPUs available for training. This parameter specifies the number of GPUs to be used during training.
    - Model path: The type of model to be trained. This parameter specifies the specific model architecture to be used.
    - Dataset location path: The dataset to be trained.
    - Swap Disk Location: The location of the swap disk. This parameter specifies the path where the swap disk is located.
    - per_device_train_batch_size: The batch size for each GPU. This parameter determines the number of training samples processed in each iteration.
    - num_train_epochs: The number of training Epochs.
    - Output Model Path: The path where the final trained model will be saved. This parameter specifies the destination folder for saving the trained model.
2. Output Objects. The SDK provides the following output:
    - Iteration Number: Current iteration count.
    - Final Model: The final trained model.