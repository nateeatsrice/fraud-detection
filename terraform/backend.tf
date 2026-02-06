terraform {
  backend "s3" {
    bucket       = "fraud-detection-tfstate-codespace-2026"
    key          = "fraud-detection/terraform.tfstate"
    region       = "us-east-2"
    use_lockfile = true
    encrypt      = true
  }
}
