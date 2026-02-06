terraform {
  backend "s3" {
    bucket         = "fraud-detection-tfstate-codespace-2026"
    key            = "fraud-detection/terraform.tfstate"
    region         = "us-east-2"
    dynamodb_table = "fraud-detection-tflock"
    encrypt        = true
  }
}
