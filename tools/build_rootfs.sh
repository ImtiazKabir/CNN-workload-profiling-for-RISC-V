#!/usr/bin/env bash
# Build a riscv64 Debian trixie rootfs with python3 + torch, without root.
# apt resolves and downloads the riscv64 packages into a private apt dir;
# the .debs are then unpacked with dpkg-deb -x (maintainer scripts are not
# run, which is fine for running python under qemu-user with -L).
#
# usage: tools/build_rootfs.sh [extra packages...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
APTDIR=$ROOT/third_party/rv-apt
ROOTFS=$ROOT/third_party/rv-rootfs
PKGS=(python3 python3-torch python3-numpy libopenblas0-pthread "$@")

mkdir -p "$APTDIR"/etc/apt/{apt.conf.d,preferences.d,trusted.gpg.d} \
         "$APTDIR"/var/lib/apt/lists/partial \
         "$APTDIR"/var/cache/apt/archives/partial \
         "$APTDIR"/var/lib/dpkg
: > "$APTDIR"/var/lib/dpkg/status
cat > "$APTDIR"/etc/apt/sources.list <<EOF
deb [arch=riscv64 signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian trixie main
deb [arch=riscv64 signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://deb.debian.org/debian trixie-updates main
deb [arch=riscv64 signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] http://security.debian.org/debian-security trixie-security main
EOF

APT=(apt-get -q
     -o Dir="$APTDIR"
     -o Dir::State::status="$APTDIR"/var/lib/dpkg/status
     -o Dir::Etc::Parts="$APTDIR"/etc/apt/apt.conf.d
     -o APT::Architecture=riscv64
     -o APT::Architectures::=riscv64
     -o APT::Install-Recommends=false
     -o Debug::NoLocking=1)
"${APT[@]}" update
"${APT[@]}" install -y --download-only "${PKGS[@]}"

rm -rf "$ROOTFS"
mkdir -p "$ROOTFS"/usr/{bin,sbin,lib}
for d in bin sbin lib; do ln -s usr/$d "$ROOTFS/$d"; done
for deb in "$APTDIR"/var/cache/apt/archives/*.deb; do
    dpkg-deb -x "$deb" "$ROOTFS"
done

# update-alternatives normally provides these (maintainer scripts don't run here).
# Default to OpenBLAS like Debian does; select a backend per run with
# LD_LIBRARY_PATH=/usr/lib/riscv64-linux-gnu/{openblas-pthread,blas:/usr/lib/riscv64-linux-gnu/lapack}.
LIBDIR=$ROOTFS/usr/lib/riscv64-linux-gnu
ln -sf openblas-pthread/libblas.so.3 "$LIBDIR"/libblas.so.3
ln -sf openblas-pthread/liblapack.so.3 "$LIBDIR"/liblapack.so.3
ln -sf openblas-pthread/libopenblas.so.0 "$LIBDIR"/libopenblas.so.0 # normally via ldconfig
echo "rootfs: $ROOTFS ($(ls "$APTDIR"/var/cache/apt/archives/*.deb | wc -l) packages)"
