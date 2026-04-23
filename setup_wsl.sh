#!/bin/bash
# CosyVoice3 WSL2 一键部署脚本
# 在 WSL2 重启后执行：bash /mnt/d/AICODE/xiaoshuo/my-daughter-is-the-god-of-wealth/setup_wsl.sh

set -e
echo "=== CosyVoice3 WSL2 部署脚本 ==="

# 0. 修复 DNS（以防万一）
echo "[Step 0] 修复 DNS..."
if ! python3 -c "import socket; socket.getaddrinfo('baidu.com', 443)" 2>/dev/null; then
    echo "DNS 不可用，尝试修复..."
    sudo bash -c 'cat > /etc/wsl.conf << EOF
[network]
generateResolvConf = false
EOF'
    sudo bash -c 'cat > /etc/resolv.conf << EOF
nameserver 223.5.5.5
nameserver 8.8.8.8
EOF'
    echo "DNS 已修复，请重新运行此脚本"
    exit 0
fi
echo "DNS 正常"

# 1. 验证 GPU
echo "[Step 1] 验证 GPU 直通..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "GPU 直通正常"
else
    echo "警告: nvidia-smi 未找到，可能需要重启后再次尝试"
fi

# 2. 安装系统依赖
echo "[Step 2] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq sox libsox-dev git git-lfs build-essential cmake
git lfs install

# 3. 初始化 conda
echo "[Step 3] 初始化 conda..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# 4. 创建 Python 环境
echo "[Step 4] 创建 cosyvoice 环境..."
if conda env list | grep -q "cosyvoice"; then
    echo "cosyvoice 环境已存在"
else
    conda create -n cosyvoice python=3.10 -y
fi
conda activate cosyvoice
echo "Python 版本: $(python --version)"

# 5. 克隆 CosyVoice
echo "[Step 5] 克隆 CosyVoice 仓库..."
COSYVOICE_DIR="$HOME/CosyVoice"
if [ -d "$COSYVOICE_DIR" ]; then
    echo "CosyVoice 目录已存在，拉取最新..."
    cd "$COSYVOICE_DIR"
    git pull
else
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"
    cd "$COSYVOICE_DIR"
    git submodule update --init --recursive
fi

# 6. 安装 Python 依赖
echo "[Step 6] 安装 Python 依赖..."
cd "$COSYVOICE_DIR"
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 7. 下载模型
echo "[Step 7] 下载 CosyVoice3 模型..."
python3 << 'PYTHON_DOWNLOAD'
from modelscope import snapshot_download
import os

models_dir = os.path.expanduser("~/CosyVoice/pretrained_models")
os.makedirs(models_dir, exist_ok=True)

print("下载 Fun-CosyVoice3-0.5B...")
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir=os.path.join(models_dir, 'Fun-CosyVoice3-0.5B')
)
print("CosyVoice3 模型下载完成")

print("下载 CosyVoice-ttsfrd 资源...")
try:
    snapshot_download(
        'iic/CosyVoice-ttsfrd',
        local_dir=os.path.join(models_dir, 'CosyVoice-ttsfrd')
    )
    print("ttsfrd 资源下载完成")
except Exception as e:
    print(f"ttsfrd 下载失败（可忽略）: {e}")
PYTHON_DOWNLOAD

# 8. 安装 ttsfrd（可选）
echo "[Step 8] 安装 ttsfrd（可选）..."
TTSFRD_DIR="$HOME/CosyVoice/pretrained_models/CosyVoice-ttsfrd"
if [ -d "$TTSFRD_DIR" ] && [ -f "$TTSFRD_DIR/resource.zip" ]; then
    cd "$TTSFRD_DIR"
    unzip -o resource.zip -d . 2>/dev/null || true
    pip install ttsfrd_dependency-0.1-py3-none-any.whl 2>/dev/null || true
    pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl 2>/dev/null || true
    echo "ttsfrd 安装完成"
else
    echo "跳过 ttsfrd（使用 wetext 替代）"
fi

# 9. 验证模型加载
echo "[Step 9] 验证模型加载..."
cd "$HOME/CosyVoice"
python3 << 'PYTHON_VERIFY'
import sys
sys.path.insert(0, 'third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel

print("加载 Fun-CosyVoice3-0.5B...")
model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
print("模型加载成功！")
print(f"可用说话人: {model.list_available_spks()}")
print(f"采样率: {model.sample_rate}")
PYTHON_VERIFY

echo ""
echo "==================================="
echo "=== 部署完成！ ==="
echo "==================================="
echo ""
echo "启动 webui 测试："
echo "  cd ~/CosyVoice"
echo "  conda activate cosyvoice"
echo "  python webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B"
echo ""
echo "浏览器访问: http://localhost:50000"
