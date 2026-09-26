variable "environment" {
  description = "Deployment environment name (dev, prod)."
  type        = string
  default     = "dev"
}

variable "region" {
  description = "Cloud provider region."
  type        = string
  default     = "us-east-1"
}

variable "gpu_instance_type" {
  description = "Compute instance size for the optional REST API / remote-inference backend."
  type        = string
  default     = "g5.xlarge"
}
