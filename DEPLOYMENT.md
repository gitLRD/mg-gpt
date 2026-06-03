# MG-GPT Lambda Deployment

Deploy MG-GPT as a serverless chat application on AWS using Lambda, API Gateway WebSocket, S3, and Terraform.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      S3 Static Website                      │
│                    (Frontend HTML/JS)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                   WebSocket
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          API Gateway WebSocket API                          │
│              (Connect/Message/Disconnect)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                   Lambda
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼────┐              ┌────────▼─────┐
   │ PyTorch │              │  DynamoDB    │
   │  Model  │              │ (Conversation│
   │(in mem) │              │  Context)    │
   └─────────┘              └──────────────┘
```

## Prerequisites

- AWS Account with appropriate permissions
- Terraform installed (`brew install terraform`)
- Docker installed
- AWS CLI configured (`aws configure`)
- Training completed: `checkpoints/ckpt_best.pt` exists

## Setup

### 1. Initialize Terraform

```bash
cd terraform
terraform init
terraform plan
```

### 2. Deploy Infrastructure

```bash
terraform apply
```

This will create:
- ECR repository
- DynamoDB table for conversation storage
- Lambda function (waiting for image)
- API Gateway WebSocket API
- S3 bucket for static website
- IAM roles and policies

Output will show:
```
websocket_endpoint = wss://abc123.execute-api.eu-west-2.amazonaws.com/prod
website_url = http://mg-gpt-chat-123456789.s3-website.eu-west-2.amazonaws.com
s3_bucket = mg-gpt-chat-123456789
ecr_repository = 123456789.dkr.ecr.eu-west-2.amazonaws.com/mg-gpt
lambda_function_name = mg-gpt-handler
```

### 3. Deploy Lambda Function & Frontend

```bash
./deploy.sh
```

This script will:
1. Build Docker image with the model
2. Push to ECR
3. Update Lambda with new image
4. Upload frontend HTML to S3

### 4. Update Frontend Config

If the WebSocket endpoint differs from what the frontend expects, update `frontend/index.html`:

```javascript
const wsUrl = `wss://YOUR_ENDPOINT.execute-api.eu-west-2.amazonaws.com/prod`;
```

Then re-run:
```bash
aws s3 cp frontend/index.html s3://YOUR_BUCKET/index.html
```

## Usage

1. Open the website URL in a browser
2. Wait for "Connected" status
3. Type a question about MG cars
4. Get response from the model

## Conversation Context

- Conversations are stored per WebSocket connection
- History maintained for the duration of the session (up to 15 minutes per Lambda timeout)
- DynamoDB TTL auto-deletes old conversations after 15 minutes
- No persistent history across sessions

## Monitoring

### View Lambda Logs

```bash
aws logs tail /aws/lambda/mg-gpt-handler --follow
```

### Check DynamoDB

```bash
aws dynamodb scan --table-name mg-gpt-conversations --region eu-west-2
```

### Monitor API Gateway

CloudWatch Logs in `API Gateway` > `WebSocket APIs` > `mg-gpt-websocket`

## Cost Estimation

Rough monthly costs (assuming 100 messages/day):
- **Lambda**: ~$0.50 (1M invocations @ $0.20/1M)
- **DynamoDB**: ~$1.25 (on-demand, ~3K requests)
- **S3**: <$1 (storage + requests)
- **Data Transfer**: ~$0.20
- **Total**: ~$3-5/month

## Troubleshooting

### Lambda cold start slow

First invocation loads the PyTorch model (~5-10s). Subsequent messages are faster.

### WebSocket connection fails

Check:
1. API Gateway endpoint is correct in frontend
2. Lambda execution role has `execute-api:ManageConnections` permission
3. Check Lambda CloudWatch logs

### Model inference errors

Check:
1. `ckpt_best.pt` was correctly copied into Docker image
2. Model config matches training config (small_config)
3. Lambda memory is sufficient (default 1024MB, should be enough)

## Cleanup

Remove all resources:

```bash
# Delete S3 bucket contents first
aws s3 rm s3://YOUR_BUCKET --recursive

# Destroy infrastructure
cd terraform
terraform destroy
```

## Customization

### Change model size

Update `terraform/main.tf`:
```hcl
memory_size = 3008  # For medium model
timeout     = 120   # Longer for bigger model
```

### Change conversation timeout

Edit `lambda/handler.py`:
```python
"ttl": int(time.time()) + 1800,  # 30 minutes instead of 15
```

### Increase API Gateway rate limit

Edit `terraform/main.tf`:
```hcl
throttle_settings {
  burst_limit = 10000
  rate_limit  = 5000
}
```

## References

- [AWS Lambda Python Runtime](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)
- [API Gateway WebSocket](https://docs.aws.amazon.com/apigateway/latest/developerguide/websockets.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
