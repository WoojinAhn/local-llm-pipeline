#!/bin/bash
# Setup script for FLUX.2 image generation via flux-2-swift-mlx.
#
# Builds Flux2CLI from source with Xcode — required because pre-built
# binaries ship without default.metallib (Metal shader library), causing
# runtime errors on Apple Silicon.
#
# Prerequisites:
#   - macOS 15+ with Apple Silicon
#   - Xcode 16+ installed (xcode-select --install)
#   - HuggingFace token for gated models (FLUX.2-dev)
#
# Usage:
#   ./setup-flux.sh              # build CLI + download Klein 4B (fast test)
#   ./setup-flux.sh --dev        # also download Dev 32B int4 (best quality)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
BUILD_DIR="/tmp/flux-2-swift-mlx"
REPO_URL="https://github.com/VincentGourbin/flux-2-swift-mlx.git"

echo "=== FLUX.2 Setup ==="

# Step 1: Metal Toolchain
echo "[1/4] Checking Metal Toolchain..."
if ! xcrun --find metal &>/dev/null; then
    echo "  Installing Metal Toolchain..."
    xcodebuild -downloadComponent MetalToolchain
else
    echo "  Metal Toolchain already installed."
fi

# Step 2: Clone source
echo "[2/4] Cloning flux-2-swift-mlx..."
if [ -d "$BUILD_DIR" ]; then
    echo "  Updating existing clone..."
    cd "$BUILD_DIR" && git pull --ff-only
else
    git clone "$REPO_URL" "$BUILD_DIR"
fi

# Step 3: Build with Xcode
echo "[3/4] Building Flux2CLI (this may take a few minutes)..."
cd "$BUILD_DIR"
xcodebuild -scheme Flux2CLI -configuration Release \
    -destination "platform=macOS" \
    -derivedDataPath build \
    -quiet

# Step 4: Copy artifacts
echo "[4/4] Installing to $BIN_DIR..."
mkdir -p "$BIN_DIR"
cp "$BUILD_DIR/build/Build/Products/Release/Flux2CLI" "$BIN_DIR/"
cp -R "$BUILD_DIR/build/Build/Products/Release/mlx-swift_Cmlx.bundle" "$BIN_DIR/"

echo ""
echo "=== Setup complete ==="
echo "  Binary: $BIN_DIR/Flux2CLI"
echo ""
echo "Quick test:"
echo "  $BIN_DIR/Flux2CLI download --model klein-4b"
echo "  $BIN_DIR/Flux2CLI t2i 'a cat in space' --model klein-4b --output outputs/test.png"

# Optional: download Dev model
if [[ "${1:-}" == "--dev" ]]; then
    echo ""
    echo "=== Downloading Dev 32B (int4) ==="
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "Error: HF_TOKEN environment variable required for gated model."
        echo "  export HF_TOKEN=hf_your_token"
        echo "  Then re-run: ./setup-flux.sh --dev"
        exit 1
    fi
    "$BIN_DIR/Flux2CLI" download --model dev --transformer-quant int4 --hf-token "$HF_TOKEN"
fi
