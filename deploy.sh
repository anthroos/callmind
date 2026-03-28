#!/bin/bash
# Deploy CallMind to a DigitalOcean Droplet
# Usage: ./deploy.sh
# Requires: doctl authenticated, .env file with secrets

set -e

DROPLET_NAME="callmind-demo"
REGION="sfo3"
SIZE="s-2vcpu-4gb"
IMAGE="docker-20-04"

echo "=== Creating DigitalOcean Droplet ==="
DROPLET_IP=$(doctl compute droplet create "$DROPLET_NAME" \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --wait \
    --format PublicIPv4 \
    --no-header)

echo "Droplet IP: $DROPLET_IP"
echo "Waiting 30s for SSH to be ready..."
sleep 30

echo "=== Deploying CallMind ==="
# Copy files
scp -o StrictHostKeyChecking=no -r \
    Dockerfile docker-compose.yml .env pyproject.toml callmind/ \
    root@"$DROPLET_IP":/root/callmind/

# Start services
ssh -o StrictHostKeyChecking=no root@"$DROPLET_IP" bash -c "'
cd /root/callmind
docker-compose up -d --build
echo \"CallMind deployed at http://$DROPLET_IP:8000\"
'"

echo ""
echo "=== DONE ==="
echo "CallMind: http://$DROPLET_IP:8000"
echo "Qdrant:   http://$DROPLET_IP:6333"
