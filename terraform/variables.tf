variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-2"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "mg-gpt"
}

variable "environment" {
  description = "Environment"
  type        = string
  default     = "prod"
}

variable "domain_name" {
  description = "Domain name for the website"
  type        = string
  default     = "mg-gpt.desforgeslabs.com"
}

variable "cloudflare_zone_id" {
  description = "Cloudflare Zone ID for desforgeslabs.com"
  type        = string
}
