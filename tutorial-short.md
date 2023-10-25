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
* Pre-configured GCE Image baked with conda and [packages for llama 2](https://github.com/saltysoup/llama-recipes/blob/main/requirements.txt) via Packer
* Autoscaling [Slurm](https://slurm.schedmd.com/documentation.html) cluster with L4/A100 GPU partitions

## Deploying the ML Cluster

### Pre-Requisite

1. You must have the following google cloud GPU quotas approved in your projects for the respective models you want to run in this tutorial
* **Llama2-7B-chat-hf** - 8 x L4-24GB GPUs (or 2 x L4-24GB GPUs for PEFT using LoRa)
* **Llama2-13B-chat-hf** - 40 x L4-24GB GPUs or 8 x A100-80GB GPUs (or 8 x L4-24GB GPUs for PEFT using LoRa)
* **Llama2-70B-chat-hf** - 16 x A100-80GB GPUs (or 8 x L4-24GB GPUs for PEFT using LoRa)

## Enable Google Cloud APIs

Enable the services for the ML Cluster by running the following command:

```bash
gcloud services enable secretmanager.googleapis.com file.googleapis.com compute.googleapis.com storage.googleapis.com serviceusage.googleapis.com 
```

If you don't have the [gcloud CLI installed]([url](https://cloud.google.com/sdk/docs/install)), enable the API by using these links.
* [Enable Compute Engine API](https://console.cloud.google.com/apis/api/compute.googleapis.com/overview?_ga=2.216372950.1572947691.1697699140-145949251.1697699140)

* [Enable Secret Manager API](https://console.cloud.google.com/apis/api/secretmanager.googleapis.com/overview?_ga=2.185357929.1572947691.1697699140-145949251.1697699140)

* [Enable Filestore API](https://console.cloud.google.com/apis/api/file.googleapis.com/overview?_ga=2.173898211.1572947691.1697699140-145949251.1697699140) 

* [Enable Cloud Storage API](https://console.cloud.google.com/apis/api/storage.googleapis.com/overview?_ga=2.245348933.1572947691.1697699140-145949251.1697699140)

* [Enable Service Usage API](https://console.cloud.google.com/apis/api/serviceusage.googleapis.com/overview?_ga=2.245348933.1572947691.1697699140-145949251.1697699140)

### Install Cloud HPC toolkit

Install the dependencies for the Cloud HPC toolkit.

Alternatively, you can use a pre-configured [cloudshell environment](https://cloud.google.com/shell/docs/launching-cloud-shell) in your GCP console.  
```bash
sudo apt-get install gcc-multilib

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

1. Create a new yaml file called `llama2-hpc.yaml` and update the placeholder values in the deployment yaml below (under `vars:`). Copy the content to your clipboard.

* `PROJECT_ID`
* `BUCKET_NAME`

```yaml
---
blueprint_name: ml-cluster

vars:
  project_id: PROJECT_NAME ## Set your project id
  deployment_name: llama2-hpc
  region: asia-southeast1 ## Set your region
  zone: asia-southeast1-c ## Set your zone
  network_name: llama-network
  subnetwork_name: llama2-subnet
  new_image_project: injae-sandbox-340804
  new_image_name: llama-slurm-20230824t114217z
  disk_size_gb: 300
  bucket_model: BUCKET_NAME ## Set your new bucket name

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
      use_deployment_name_in_bucket_name: false

  - id: bucket_mount
    source: modules/file-system/pre-existing-network-storage
    settings:
      remote_mount: $(vars.bucket_model)
      local_mount: /bucket
      fs_type: gcsfuse
      mount_options: defaults,_netdev,implicit_dirs,allow_other,dir_mode=0777,file_mode=766

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
        name: $(vars.new_image_name)
        project: $(vars.new_image_project)
      instance_image_custom: true

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
        name: $(vars.new_image_name)
        project: $(vars.new_image_project)
      instance_image_custom: true

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
    - g2_partition
    - homefs
    - bucket_mount
    settings:
      disable_controller_public_ips: false
      instance_image:
        name: $(vars.new_image_name)
        project: $(vars.new_image_project)
      instance_image_custom: true

  - id: slurm_login
    source: community/modules/scheduler/schedmd-slurm-gcp-v5-login
    use:
    - network1
    - slurm_controller
    settings:
      instance_image:
        name: $(vars.new_image_name)
        project: $(vars.new_image_project)
      instance_image_custom: true
```

2. Save the updated contents into the `llama2_hpc.yaml` file. *Reminder: Don't forget to include your project id, region, zone, bucket name*

3. Use the `ghpc` tool to generate the deployment files
    ```bash
    ./ghpc create llama2-hpc.yaml
    ```

4. Use the `ghpc` tool to deploy the cloud resources
    ```bash
    ./ghpc deploy llama2-hpc
    ```
 
    *When prompted during deployment, enter "a" to apply and continue deploying.*

    Note: Deployment across all groups should take approx. 10 min to complete.

## Running training and inference jobs

Once the deployment is complete, you should now have a complete AI infrastructure platform to begin our training jobs. This includes a new slurm cluster that will automatically scale up the GPU nodes for incoming jobs and scale back down once the jobs are finished running.

This will ensure that GPU nodes are only run for the duration of the tasks and does not incur costs when there is no active jobs in the queue.

### Getting started with fine tuning

1. Access the Google cloud console on your browser and navigate to the [Compute Engine portal](https://console.cloud.google.com/compute/instances). You should see 1 x Slurm controller VM and 1 x Slurm login node VM.

2. SSH into the Slurm login node by clicking on the SSH button next to the VM on the console. The VM will be called `llama2hpc-login-*`.

3. Within the shell session of the login node,  activate the working environment with
    ```bash
    conda activate llama
    ```

4. Clone the example repo, which contains training and inference scripts

    ```bash
    git clone -b bootcamp --single-branch https://github.com/saltysoup/llama-recipes.git

    cd llama-recipes/examples

    ```

5. Open the `single_node.slurm` script and review how we can pass dynamic values for multi-nodes and multi-gpus from Slurm to torchrun. 

6. Submit the training job to slurm
    ```bash
    sbatch single_node.slurm
    ```

7. Slurm will now create a new G2/A2 VM to run the training task. Once the job is complete, the VM will automatically be destroyed.

8. To check the status of slurm jobs
    ```bash
    watch squeue
    ```
    *Note: Full fine tuning Llama-2-7b-chat-hf with 8 x L4 GPUs using bf16 datatype takes approx. 10 min to complete. The training dataset is a small subset of [samsum on HF](https://huggingface.co/datasets/samsum)*

    ** Interested in finetuning the entire dataset? modify the [datasets.py script](https://github.com/saltysoup/llama-recipes/blob/main/examples/llama_recipes/configs/datasets.py) to use `train_split: str = "train"` **

9. Once the job is complete, Slurm will automatically create a new file (slurm-1.out) with contents from the training node's stdout/stderr output.    

### Getting started with inferencing

1. To load the model for inferencing, we first need to convert the FSDP sharded checkpoints into HuggingFace checkpoints.

    ```bash
    sbatch convert_checkpoints.slurm
    ```
   ** This takes approximately 10 min to complete **

This will generate another output file eg. slurm-2.out. View the contents to see where the converted checkpoint is stored.

2. Once this is complete, you can load the model interactively or with an example job.
    ```bash
    sbatch test_inference.slurm
    ```

This script will use a sample prompt in `chat_completion/chats.json` to generate a summary of the dialogue, using the model we fine tuned and converted previously.

3. Viewing the output of the `test_inference.slurm` job, you should see an output similar to below:

```
  the inference time is 14243.336016000001 ms
User input and model output deemed safe.
Model output:
{

        {

                "role": "system",

                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."

        },

        {

                "role": "user",

                "content": "Summarize this conversation. John: My favourite food in the world is a cheese pizza. What about you?  Jane: I like to have a nice cold glass of water.  John: That's not really a food, tell me another one!  Jane: Fine. I like to photosynthesise in the Sun. I'm actually a plant. John: We need to get you to a doctor ASAP.  Jane: OK. I'll make an appointment at tree thirty PM. John: LOL how funny. "

        }

}
!
Summary:
John and Jane like cheese pizza and photosynthesising in the Sun. John needs to get Jane to a doctor's appointment at tree thirty PM. 
```

## Cleaning up

To destroy the cloud environment we created, use the following command.

```bash
cd hpc-toolkit
./ghpc destroy llama2_hpc
```

*When prompted during the environment tear down, enter "a" to confirm.*
