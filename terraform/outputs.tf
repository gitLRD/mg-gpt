output "websocket_endpoint" {
  description = "WebSocket API endpoint"
  value       = aws_apigatewayv2_stage.websocket.invoke_url
}

output "website_url" {
  description = "S3 website URL"
  value       = "http://${aws_s3_bucket.website.id}.s3-website.${var.aws_region}.amazonaws.com"
}

output "s3_bucket" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.website.id
}

output "ecr_repository" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.lambda_image.repository_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.websocket_handler.function_name
}

output "dynamodb_table" {
  description = "DynamoDB table name"
  value       = aws_dynamodb_table.conversations.name
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain"
  value       = aws_cloudfront_distribution.website.domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (for cache invalidations)"
  value       = aws_cloudfront_distribution.website.id
}

output "website_domain" {
  description = "Custom domain for website"
  value       = var.domain_name
}
