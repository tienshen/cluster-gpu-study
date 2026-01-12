# Cloud Infrastructure Setup

Quick reference for deploying GPU instances across major cloud providers.

## AWS GPU Instances

### Launch with AWS CLI
```bash
aws ec2 run-instances \
    --image-id ami-0c2b8ca1dad447f8a \
    --instance-type g4dn.xlarge \
    --key-name YOUR_KEY \
    --security-group-ids sg-xxxxx \
    --iam-instance-profile Name=YOUR_PROFILE \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=triton-benchmark}]'
```

### Connect
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>
```

### Recommended Instance Types
- **g4dn.xlarge**: 1x T4 GPU, 4 vCPU, 16GB RAM (~$0.526/hr)
- **g5.xlarge**: 1x A10G GPU, 4 vCPU, 16GB RAM (~$1.006/hr)
- **g5.2xlarge**: 1x A10G GPU, 8 vCPU, 32GB RAM (~$1.212/hr)

### Spot Instances (70% discount)
```bash
aws ec2 request-spot-instances \
    --spot-price "0.20" \
    --instance-count 1 \
    --launch-specification file://spot-config.json
```

---

## GCP GPU Instances

### Launch with gcloud
```bash
gcloud compute instances create triton-benchmark \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

### Connect
```bash
gcloud compute ssh triton-benchmark --zone=us-central1-a
```

### Recommended Configurations
- **T4**: `--accelerator=type=nvidia-tesla-t4,count=1` (~$0.35/hr + VM)
- **A100**: `--accelerator=type=nvidia-tesla-a100,count=1` (~$2.93/hr + VM)

### Preemptible (80% discount)
```bash
gcloud compute instances create triton-benchmark \
    --preemptible \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    ...
```

---

## Azure GPU Instances

### Launch with Azure CLI
```bash
az vm create \
    --resource-group triton-rg \
    --name triton-benchmark \
    --location eastus \
    --size Standard_NC4as_T4_v3 \
    --image "microsoft-dsvm:ubuntu-2004:ubuntu-2004:latest" \
    --admin-username azureuser \
    --generate-ssh-keys \
    --public-ip-sku Standard \
    --os-disk-size-gb 100
```

### Connect
```bash
ssh azureuser@<public-ip>
```

### Recommended VM Sizes
- **NC4as_T4_v3**: 1x T4 GPU, 4 vCPU, 28GB RAM (~$0.526/hr)
- **NC6s_v3**: 1x V100 GPU, 6 vCPU, 112GB RAM (~$3.06/hr)
- **NC24ads_A100_v4**: 1x A100 GPU, 24 vCPU, 220GB RAM (~$3.67/hr)

### Spot Instances
```bash
az vm create \
    --priority Spot \
    --max-price 0.20 \
    --eviction-policy Deallocate \
    ...
```

---

## Post-Launch Setup (All Providers)

### 1. Verify GPU
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 2. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/cluster-gpu-study.git
cd cluster-gpu-study
git checkout triton-overhaul
```

### 3. Pull Docker Images
```bash
docker pull nvcr.io/nvidia/tensorrt_llm/release:v0.7.1
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
```

---

## Terraform Example

Save as `main.tf`:
```hcl
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "triton" {
  ami           = "ami-0c2b8ca1dad447f8a"
  instance_type = "g4dn.xlarge"
  key_name      = var.key_name

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "triton-benchmark"
  }

  user_data = <<-EOF
    #!/bin/bash
    cd /home/ubuntu
    git clone https://github.com/YOUR_USERNAME/cluster-gpu-study.git
  EOF
}

output "instance_ip" {
  value = aws_instance.triton.public_ip
}
```

Deploy:
```bash
terraform init
terraform apply
terraform destroy  # when done
```

---

## Cost Monitoring

### AWS
```bash
aws ce get-cost-and-usage \
    --time-period Start=2026-01-01,End=2026-01-07 \
    --granularity DAILY \
    --metrics BlendedCost
```

### GCP
```bash
gcloud billing accounts list
gcloud beta billing budgets list --billing-account=ACCOUNT_ID
```

### Azure
```bash
az consumption usage list \
    --start-date 2026-01-01 \
    --end-date 2026-01-07
```
