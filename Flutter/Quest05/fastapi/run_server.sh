#!/bin/bash

# 첫 번째 인자를 main 파일 이름으로 사용
MAIN_FILE=$1

if [ -z "$MAIN_FILE" ]; then
    echo "Error: Please provide the main file name as an argument."
    echo "Usage: ./run_server.sh <main_file_name>"
    exit 1
fi

poetry run uvicorn ${MAIN_FILE}:app \
    --host 0.0.0.0 \
    --port 8443 \
    --ssl-keyfile /Users/oneyoon/Study/ssl_certificate/playcode_key.pem \
    --ssl-certfile /Users/oneyoon/Study/ssl_certificate/playcode_cert.pem

