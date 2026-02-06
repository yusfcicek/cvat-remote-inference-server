#!/bin/bash
# CVAT Model Deployment Script
# Usage: ./deploy.sh [target_directory]

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TARGET_DIR=${1:-$SCRIPT_DIR}

export DOCKER_BUILDKIT=1

# 1. Ensure the base image exists
if [[ "$(docker images -q cvat.openvino.base 2> /dev/null)" == "" ]]; then
    echo "[Deploy] Building CVAT OpenVINO base image..."
    docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"
fi

# 2. Ensure CVAT project exists
nuctl create project cvat --platform local || true

# 3. Find and deploy function(s)
find "$TARGET_DIR" -name "function.yaml" | while read -r func_config; do
    func_root="$(dirname "$func_config")"
    func_name="$(basename "$func_root")"
    
    # Generic templates use 'template' or 'yolov12n' as name, we ensure it's the folder name
    echo "[Deploy] Deploying function: $func_name..."

    nuctl deploy "$func_name" \
        --project-name cvat \
        --path "$func_root" \
        --file "$func_config" \
        --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

# 4. Status summary
echo "[Deploy] Current functions on local platform:"
nuctl get function --platform local
