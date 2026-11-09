"""Create private bucket and retention rules; use only with an owned application bucket."""

import os
from botocore.exceptions import ClientError
from worker.main import object_store

s3 = object_store()
bucket = os.getenv("S3_BUCKET", "watermark")
try:
    s3.head_bucket(Bucket=bucket)
except ClientError as e:
    if str(e.response["Error"]["Code"]) not in ("404", "NoSuchBucket"):
        raise
    region = os.getenv("S3_REGION", "us-east-1")
    args = {"Bucket": bucket}
    if region != "us-east-1":
        args["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**args)
s3.put_public_access_block(
    Bucket=bucket,
    PublicAccessBlockConfiguration={
        "BlockPublicAcls": True,
        "IgnorePublicAcls": True,
        "BlockPublicPolicy": True,
        "RestrictPublicBuckets": True,
    },
)
s3.put_bucket_lifecycle_configuration(
    Bucket=bucket,
    LifecycleConfiguration={
        "Rules": [
            {
                "ID": "intermediate-24h",
                "Status": "Enabled",
                "Filter": {"Tag": {"Key": "kind", "Value": "intermediate"}},
                "Expiration": {"Days": 1},
            },
            {
                "ID": "results-7d",
                "Status": "Enabled",
                "Filter": {"Tag": {"Key": "kind", "Value": "result"}},
                "Expiration": {"Days": 7},
            },
            {
                "ID": "unreferenced-7d",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Expiration": {"Days": 7},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            },
        ]
    },
)
print("Private application bucket and lifecycle rules configured.")
