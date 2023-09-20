# Fine-tuning Llama2 on Google Cloud with the Cloud HPC toolkit
With growing public access to bigger and more capable Large Language Models (LLMs) such as Llama2 and Falcon, we're also seeing a growing trend of customers looking to adopt and customise (fine-tune) these foundational models for their specific use cases with custom datasets. By fine tuning these foundational models, customers can quickly turn these general purpose language models into domain experts;- allowing the models to respond with specific responses to user prompts and reduce the chance of hallucinated responses.

As recent LLMs are trained with billions of parameters, training such models require a highly scalable and performance optimised AI Infrastructure platform, through distributed training spanning multiple GPUs across multiple nodes.

To help customers accelerate their large scale ML training, we'll be exploring how you can use the Cloud HPC toolkit to design, optimise and deploy a complete AI infrastructure platform on Google Cloud. Although this codelab will focus on Llama 2 as the example workload, a similar approach can be utilised for other demanding ML training spanning multi nodes.

Alternatively, if you're looking to deploy or fine tune Llama 2 with PEFT, check out [Llama 2 on Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/139)

## Solution Overview
In this codelab, we will be using the [Cloud HPC toolkit](https://cloud.google.com/hpc-toolkit/docs/overview) to help design, optimise and deploy our ML cluster to run our distributed training. The HPC toolkit deployment blueprint will automatically configure and deploy the following components:

* VPC with firewall configured for Cloud IAP to allow [OS Login](https://cloud.google.com/compute/docs/oslogin) and inter-node TCP commmunication
* [Filestore](https://cloud.google.com/filestore) for shared NFS storage mounted automatically and used by the cluster
* GCS bucket mounted automatically via [GCS fuse](https://cloud.google.com/storage/docs/gcs-fuse) by the cluster
* New GCE Image baked with conda and [packages for llama 2](https://github.com/saltysoup/llama-recipes/blob/main/requirements.txt) with Packer
* Autoscaling [Slurm](https://slurm.schedmd.com/documentation.html) cluster with L4/A100 GPU partitions

## Deploying the GPU Cluster

### Pre-Requisite
You must have the following google cloud GPU quotas approved in your projects for the respective models you want to run in this codelab
* **Llama2-7B-chat-hf** - 8 x L4-24GB GPUs
* **Llama2-13B-chat-hf** - 40 x L4-24GB GPUs or 8 x A100-80GB GPUs
* **Llama2-70B-chat-hf** - 16 x A100-80GB GPUs

### Install Cloud HPC toolkit

Install the dependencies for the Cloud HPC toolkit
```bash
# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform -y

#Install Packer
sudo apt install packer -y

#Install Make
sudo apt-get install make -y

#Install Go
cd /tmp
wget https://go.dev/dl/go1.21.1.linux-386.tar.gz -q
sudo tar -C /usr/local -xzf go1.21.1.linux-386.tar.gz

export PATH=$PATH:/usr/local/go/bin
export PATH=$PATH:$(go env GOPATH)/bin
go version
```

### Installing Cloud HPC toolkit
After cloning the Cloud HPC toolkit repo, compile the ghpc tool to create and deploy cloud resources using blueprints.
```bash
git clone https://github.com/GoogleCloudPlatform/hpc-toolkit.git
cd hpc-toolkit
make
```

### Configure the deployment blueprint for Cloud HPC toolkit to deploy a GPU training cluster

1. Update the placeholder values in the deployment yaml below (under `vars:`) and copy the content to your clipboard.

* `PROJECT_ID`
* `REGION`
* `ZONE`
* `BUCKET_NAME`

    ```yaml
    ---
    blueprint_name: ml-cluster

    vars:
    project_id: PROJECT_NAME ## Set your project id
    deployment_name: llama2_hpc
    region: REGION ## Set your region
    zone: ZONE ## Set your zone
    network_name: llama-network
    subnetwork_name: llama2-subnet
    new_image_family: llama2-slurm
    disk_size_gb: 200
    bucket_model: BUCKET_NAME ## Set your bucket name

    deployment_groups:
    - group: setup
    modules:
    - id: network1
        source: modules/network/vpc
    - id: homefs
        source: modules/file-system/filestore
        use:
        - network1
        settings:
        local_mount: /home
    - id: bucket_create
        source: community/modules/file-system/cloud-storage-bucket
        settings:
        name_prefix: $(vars.bucket_model)
    - id: bucket_mount
        source: modules/file-system/pre-existing-network-storage
        settings:
        remote_mount: $(vars.bucket_model)
        local_mount: /bucket
        fs_type: gcsfuse
        mount_options: defaults,_netdev,implicit_dirs,allow_other,dir_mode=0777,file_mode=766
    - id: script
        # configure conda environment for llama
        source: modules/scripts/startup-script
        settings:
        runners:
        - type: shell
            destination: install-ml-libraries.sh
            content: |
            #!/bin/bash
            # this script is designed to execute on Slurm images published by SchedMD that:
            # - are based on Debian 11 distribution of Linux
            # - have NVIDIA Drivers v530 pre-installed
            # - have CUDA Toolkit 12.1 pre-installed.

            set -e -o pipefail

            echo "deb https://packages.cloud.google.com/apt google-fast-socket main" > /etc/apt/sources.list.d/google-fast-socket.list
            apt-get update
            apt-get install --assume-yes google-fast-socket

            CONDA_BASE=/opt/conda

            if [ -d $CONDA_BASE ]; then
                    exit 0
            fi

            DL_DIR=$(mktemp -d)
            cd $DL_DIR
            curl -O https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
            HOME=$DL_DIR bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p $CONDA_BASE

            cd -
            rm -rf $DL_DIR
            unset DL_DIR

            source $CONDA_BASE/bin/activate base
            conda init --system
            # following channel ordering is important! use strict_priority!
            conda config --system --set channel_priority strict
            conda config --system --remove channels defaults
            conda config --system --add channels conda-forge
            conda config --system --add channels nvidia
            conda config --system --add channels nvidia/label/cuda-12.1.0

            pip install nvidia-cudnn-cu12 nvidia-nccl-cu12
            conda update -n base conda --yes

            cd $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib/
            ln -s libnccl.so.2 libnccl.so
            cd -

            mkdir -p $CONDA_PREFIX/etc/conda/activate.d
            echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
            echo 'NVIDIA_PYTHON_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
            echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$NVIDIA_PYTHON_PATH/cudnn/lib/:$NVIDIA_PYTHON_PATH/nccl/lib/:$CONDA_PREFIX/envs/llama/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

            mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
            echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
            echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

            ### create a virtual environment for llama
            conda create -n llama python=3.10 --yes
            conda activate llama
            conda config --env --add channels llama
            conda install -n llama appdirs=1.4.4 loralib=0.1.1 cuda=12.1.0 black=23.7.0 black-jupyter=23.7.0 py7zr=0.20.6 scipy=1.11.1 optimum=1.1.1 datasets=2.14.4 accelerate=0.21.0 peft=0.3.0 bitsandbytes=0.41.0 fairscale=0.4.13 fire=0.5.0 sentencepiece=0.1.99 transformers=4.31.0 --yes
            pip install trl==0.4.7
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

            # Compile bitsandbytes from source for cuda 12.1 support
            git clone https://github.com/TimDettmers/bitsandbytes.git
            cd bitsandbytes
            export CUDA_HOME=/usr/local/cuda-12.1 && make cuda12x CUDA_VERSION=121
            export CUDA_HOME=/usr/local/cuda-12.1 && make cuda12x_nomatmul CUDA_VERSION=121
            CUDA_VERSION=121 && /opt/conda/bin/python3.10 setup.py install
            cp bitsandbytes/libbitsandbytes_cuda121* /opt/conda/envs/llama/lib/python3.10/site-packages/bitsandbytes/
            ln -sf /opt/conda/envs/llama/lib/libstdc++.so.6.0.31 /lib/x86_64-linux-gnu/libstdc++.so.6.0.28

    - group: packer
    modules:
    - id: custom-image
        source: modules/packer/custom-image
        kind: packer
        use:
        - network1
        - script
        settings:
        source_image_project_id: [schedmd-slurm-public]
        source_image_family: slurm-gcp-5-7-debian-11
        disk_size: $(vars.disk_size_gb)
        image_family: $(vars.new_image_family)
        machine_type: c2-standard-8 # building this image does not require a GPU-enabled VM
        state_timeout: 30m

    - group: cluster
    modules:
    - id: ssd-startup
        source: modules/scripts/startup-script
        settings:
        runners:
        - type: shell
            destination: "/tmp/mount_ssd.sh"
            content: |
            #!/bin/bash
            set -ex
            export LOG_FILE=/tmp/ssd-setup.log
            export DST_MNT="/scratch"
            if [ -d $DST_MNT ]; then
                echo "DST_MNT already exists. Canceling." >> $LOG_FILE
                exit 0
            fi
            apt -y install mdadm
            lsblk >> $LOG_FILE
            export DEVICES=`lsblk -d -n -oNAME,RO | grep 'nvme.*0$' | awk {'print "/dev/" $1'}`
            mdadm --create /dev/md0 --level=0 --raid-devices=8 $DEVICES
            mkfs.ext4 -F /dev/md0
            mkdir -p $DST_MNT
            mount /dev/md0 $DST_MNT
            chmod a+w $DST_MNT
            echo UUID=`blkid -s UUID -o value /dev/md0` $DST_MNT ext4 discard,defaults,nofail 0 2 | tee -a /etc/fstab
            cat /etc/fstab >> $LOG_FILE
            echo "DONE" >> $LOG_FILE

    - id: a2_node_group
        source: community/modules/compute/schedmd-slurm-gcp-v5-node-group
        settings:
        node_count_dynamic_max: 2
        bandwidth_tier: gvnic_enabled
        disable_public_ips: false
        enable_smt: true
        machine_type: a2-ultragpu-8g
        disk_type: pd-ssd
        disk_size_gb: $(vars.disk_size_gb)
        on_host_maintenance: TERMINATE
        instance_image:
            family: $(vars.new_image_family)
            project: $(vars.project_id)

    - id: a2_partition
        source: community/modules/compute/schedmd-slurm-gcp-v5-partition
        use:
        - ssd-startup
        - a2_node_group
        - homefs
        - bucket_mount
        - network1
        settings:
        zone: $(vars.zone)
        partition_name: a100
        enable_placement: true

    - id: g2_node_group
        source: community/modules/compute/schedmd-slurm-gcp-v5-node-group
        settings:
        node_count_dynamic_max: 5
        bandwidth_tier: gvnic_enabled
        disable_public_ips: false
        enable_smt: true
        machine_type: g2-standard-96
        disk_type: pd-ssd
        disk_size_gb: $(vars.disk_size_gb)
        on_host_maintenance: TERMINATE
        instance_image:
            family: $(vars.new_image_family)
            project: $(vars.project_id)

    - id: g2_partition
        source: community/modules/compute/schedmd-slurm-gcp-v5-partition
        use:
        - g2_node_group
        - homefs
        - bucket_mount
        - network1
        settings:
        partition_name: l4
        enable_placement: false # enable compact placement policy
        partition_startup_scripts_timeout: 1200

    - id: slurm_controller
        source: community/modules/scheduler/schedmd-slurm-gcp-v5-controller
        use:
        - network1
        - a2_partition
        - homefs
        - bucket_mount
        settings:
        disable_controller_public_ips: false
        instance_image:
            family: $(vars.new_image_family)
            project: $(vars.project_id)

    - id: slurm_login
        source: community/modules/scheduler/schedmd-slurm-gcp-v5-login
        use:
        - network1
        - slurm_controller
        settings:
        instance_image:
            family: $(vars.new_image_family)
            project: $(vars.project_id)
    ```

2. Create a new yaml file called `llama2_hpc.yaml` and copy the updated contents into the file. *Reminder: Don't forget to include your project id, region, zone, bucket name*

3. Use the `ghpc` tool to generate the deployment files
    ```bash
    ./ghpc create llama2_hpc.yaml
    ```

4. Use the `ghpc` tool to deploy the cloud resources
    ```bash
    ./ghpc deploy llama2_hpc
    ```
 
    *When prompted during deployment, enter "a" to apply and continue deploying.*

    Note: the Packer deployment step takes approx ~20 min to complete as it needs to download and install pip/conda packages before creating a new GCE image for the slurm nodes.

## Running training and inference jobs

Once the deployment is complete, you should now have a complete AI infrastructure platform to begin our training jobs. This includes a new slurm cluster that will automatically scale up the GPU nodes for incoming jobs and scale back down once the jobs are finished running.

This will ensure that GPU nodes are only run for the duration of the tasks and does not incur costs when there is no active jobs in the queue.

### Getting started with fine tuning

1. Access the Google cloud console on your browser and navigate to the [Compute Engine portal](https://console.cloud.google.com/compute/instances). You should see 1 x Slurm controller VM and 1 x Slurm login node VM.

2. SSH into the Slurm login node by clicking on the SSH button next to the VM on the console.

3. Within the shell session of the login node,  activate the working environment with
    ```bash
    conda activate llama
    ```

4. Clone the example repo, which contains training and inference scripts

    ```bash
    git clone https://github.com/saltysoup/llama-recipes.git

    cd llama-recipes/examples
    ```

5. Review and modify the `multi_node.slurm` script. 

    5.1 You will need to add your [Huggingface User access token](https://huggingface.co/docs/hub/security-tokens) in the script `HUGGINGFACE_TOKEN="<YOUR TOKEN>"`. If you don't have a HuggingFace account, [create one now](https://huggingface.co/login). 

    5.2 To change the number of nodes, change the value for `SBATCH --nodes=` at the top of the script. To change between G2 VMs (L4-24GB GPU) and A2 VMs (A100-80GB GPU), change the values for `SBATCH --partition=l4/a100`.

   **Each G2 and A2 VM is pre-configured with 8 x respective GPUs each.**
   
    * To use the **Llama-2-7b-chat-hf** model, you will need at least 8 x L4-24GB GPUs. eg. `--nodes=1` and `--partition=l4`
    * To use the **Llama-2-13b-chat-hf** model, you will need at least 40 x L4-24GB GPUs or 8 x A100-80GB GPUs
    * To use the **Llama-2-70b-chat-hf** model, you will need at least 16 x A100-80GB GPUs

        *Don't have enough GPUs for larger models? lower the amount of GPUs required by using these parameters in the training script ```
        ```bash
        time srun torchrun .. finetuning.py --use_peft --peft_method lora --quantization
        ```

7. Submit the training job to slurm
    ```bash
    sbatch multi_node.slurm
    ```
8. To check the status of slurm jobs
    ```bash
    watch squeue
    ```
    *Note: Full fine tuning Llama-2-7b-chat-hf with 8 x L4 GPUs using bf16 datatype takes approx. 3 hours to complete.*
    
    *Llama-2-13b-chat-hf with 40 x L4 GPUs takes approx. 2 hours.*
    
    *Llama-2-70b-chat-hf with 16 x A100-80GB GPUs takes approx. 3 hours to complete.*

### Getting started with inferencing

1. To load the model for inferencing, we first need to convert the FSDP sharded checkpoints into HuggingFace checkpoints.

    ```bash
    sbatch convert_checkpoints.slurm
    ```

2. Once this is complete, you can load the model interactively or with an example job.
    ```bash
    sbatch test_inference.slurm
    ```