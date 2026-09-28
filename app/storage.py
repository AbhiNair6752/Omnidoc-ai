import os
import boto3

S3_BUCKET = os.environ.get("S3_BUCKET")

s3 = boto3.client(
    "s3",
    region_name=os.environ.get("AWS_REGION", "ap-south-1"),
    endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
)

def upload_file(local_path, key):
    s3.upload_file(local_path, S3_BUCKET, key)
    return key

def download_file(key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(S3_BUCKET, key, local_path)
    return local_path