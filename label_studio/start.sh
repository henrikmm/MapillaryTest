#!/bin/bash
cd "$(dirname "$0")/.."

# Mata processos anteriores se ainda estiverem rodando
pkill -f "http.server 8081" 2>/dev/null
pkill -f "label-studio start" 2>/dev/null
sleep 1

# Servidor de imagens em background
python3 label_studio/image_server.py 8081 &
IMAGE_PID=$!
echo "Image server PID $IMAGE_PID → http://localhost:8081"

# Encerra o servidor de imagens quando o Label Studio parar
trap "kill $IMAGE_PID 2>/dev/null" EXIT

# Label Studio em foreground
label-studio start --port 8080
