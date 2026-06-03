#!/bin/bash
set -e

echo "🚀 MG-GPT Lambda Deployment"
echo "=============================="

# Config
AWS_REGION="eu-west-2"
PROJECT_NAME="mg-gpt"

# Cloudflare token for terraform provider (DNS records, cert validation).
# Falls back to existing CLOUDFLARE_API_TOKEN env var if already set.
CF_TOKEN_FILE="${HOME}/.config/cloudflare/token"
if [ -z "${CLOUDFLARE_API_TOKEN:-}" ] && [ -f "${CF_TOKEN_FILE}" ]; then
  export CLOUDFLARE_API_TOKEN="$(tr -d '[:space:]' < "${CF_TOKEN_FILE}")"
fi
if [ -z "${CLOUDFLARE_API_TOKEN:-}" ]; then
  echo "❌ CLOUDFLARE_API_TOKEN not set and ${CF_TOKEN_FILE} not found"
  exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"

echo "📦 Building Docker image (x86_64 for Lambda)..."
# Lambda requires Docker manifest v2 schema 2, not OCI format.
# --provenance=false strips the attestation manifest (otherwise buildx emits an OCI index).
# --output type=docker forces the Docker v2 manifest and loads into the local daemon.
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --output type=docker \
  -f lambda/Dockerfile \
  -t ${ECR_REPO}:latest .

echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "📤 Pushing image to ECR..."
docker push ${ECR_REPO}:latest

echo "✅ Docker image pushed"

# Apply remaining Terraform (Lambda + routes)
echo "🔄 Applying Terraform for Lambda function..."
cd terraform
terraform apply -auto-approve
LAMBDA_NAME=$(terraform output -raw lambda_function_name)
cd ..

# Terraform won't redeploy the Lambda when only the image content changes
# (image_uri tag stays ":latest"). Force a code update from the latest digest.
echo "🔄 Updating Lambda to latest image digest..."
aws lambda update-function-code \
  --function-name "${LAMBDA_NAME}" \
  --image-uri "${ECR_REPO}:latest" \
  --publish \
  --region "${AWS_REGION}" > /dev/null
aws lambda wait function-updated \
  --function-name "${LAMBDA_NAME}" \
  --region "${AWS_REGION}"

echo "✅ Lambda function deployed"

# Upload frontend to S3 (substitute WebSocket endpoint into HTML first)
echo "📤 Uploading frontend to S3..."
S3_BUCKET=$(cd terraform && terraform output -raw s3_bucket)
WS_ENDPOINT_FOR_HTML=$(cd terraform && terraform output -raw websocket_endpoint)
TMP_HTML=$(mktemp -t mg-gpt-index.XXXXXX.html)
sed "s|__WS_ENDPOINT__|${WS_ENDPOINT_FOR_HTML}|g" frontend/index.html > "${TMP_HTML}"
aws s3 cp "${TMP_HTML}" s3://${S3_BUCKET}/index.html \
  --content-type "text/html" \
  --cache-control "max-age=3600" \
  --region ${AWS_REGION}
rm -f "${TMP_HTML}"

# Invalidate CloudFront cache so the new HTML is served immediately
CF_DIST_ID=$(cd terraform && terraform output -raw cloudfront_distribution_id)
echo "🔄 Invalidating CloudFront cache..."
aws cloudfront create-invalidation --distribution-id "${CF_DIST_ID}" --paths "/index.html" > /dev/null

echo "✅ Frontend uploaded"

# Get endpoints
WEBSITE_DOMAIN=$(cd terraform && terraform output -raw website_domain)
WS_ENDPOINT=$(cd terraform && terraform output -raw websocket_endpoint)

echo ""
echo "🎉 Deployment complete!"
echo "=============================="
echo "🌐 Website:  https://${WEBSITE_DOMAIN}"
echo "🔌 WebSocket: ${WS_ENDPOINT}"
echo ""
echo "DNS may take 5-10 minutes to propagate."
echo "Frontend will auto-connect to the WebSocket endpoint."
