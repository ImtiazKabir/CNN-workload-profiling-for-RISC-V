#!/usr/bin/env bash
# Build qemu-riscv64 (linux-user) with TCG plugin support into third_party/qemu.
set -euo pipefail
VER=${QEMU_VERSION:-10.0.3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=$ROOT/third_party
mkdir -p "$DEST"
cd "$DEST"
if [ ! -d qemu ]; then
    [ -f qemu-$VER.tar.xz ] || curl -fL -o qemu-$VER.tar.xz https://download.qemu.org/qemu-$VER.tar.xz
    tar xf qemu-$VER.tar.xz
    mv qemu-$VER qemu
fi
cd qemu
if [ ! -f build/build.ninja ]; then
    ./configure --target-list=riscv64-linux-user --enable-plugins \
        --disable-system --disable-docs --disable-tools --disable-werror
fi
ninja -C build qemu-riscv64
echo "built: $DEST/qemu/build/qemu-riscv64"
