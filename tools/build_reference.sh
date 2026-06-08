#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT/third_party/reference"
COMMIT="3461d9d3b8d31baa0eb35e5acc60bedd283f1dd3"
SRC="$REF_DIR/pariasm-vnlb"
BUILD="$REF_DIR/build/pariasm-vnlb"
PATCH_FILE="$REF_DIR/patches/pariasm-vnlb-macos-build.patch"

OPENBLAS_ROOT="${OPENBLAS_ROOT:-/opt/homebrew/opt/openblas}"
PNG_INCLUDE="${PNG_INCLUDE:-/opt/homebrew/opt/libpng/include}"
JPEG_INCLUDE="${JPEG_INCLUDE:-/opt/homebrew/opt/jpeg-turbo/include}"
TIFF_INCLUDE="${TIFF_INCLUDE:-/opt/homebrew/opt/libtiff/include}"

mkdir -p "$REF_DIR"

if [ ! -d "$SRC/.git" ]; then
	git clone https://github.com/pariasm/vnlb "$SRC"
	git -C "$SRC" checkout "$COMMIT"
fi

CURRENT=$(git -C "$SRC" rev-parse HEAD)
if [ "$CURRENT" != "$COMMIT" ]; then
	echo "reference source is at $CURRENT, expected $COMMIT" >&2
	exit 1
fi

if git -C "$SRC" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
	:
else
	git -C "$SRC" apply "$PATCH_FILE"
fi

cmake \
	-S "$SRC" \
	-B "$BUILD" \
	-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
	-DCMAKE_BUILD_TYPE=Release \
	-DCBLAS_INCLUDES:PATH="$OPENBLAS_ROOT/include" \
	-DCBLAS_LIBRARIES:FILEPATH="$OPENBLAS_ROOT/lib/libopenblas.dylib" \
	-DLAPACKE_INCLUDES:PATH="$OPENBLAS_ROOT/include" \
	-DLAPACKE_LIB:FILEPATH="$OPENBLAS_ROOT/lib/libopenblas.dylib" \
	-DLAPACK_LIB:FILEPATH="$OPENBLAS_ROOT/lib/libopenblas.dylib" \
	-DBLAS_LIBRARIES:FILEPATH="$OPENBLAS_ROOT/lib/libopenblas.dylib" \
	-DPNG_PNG_INCLUDE_DIR:PATH="$PNG_INCLUDE" \
	-DJPEG_INCLUDE_DIR:PATH="$JPEG_INCLUDE" \
	-DTIFF_INCLUDE_DIR:PATH="$TIFF_INCLUDE"

cmake --build "$BUILD" --parallel

if [ "$(uname)" = "Darwin" ]; then
	LLVM_LIBOMP="/opt/homebrew/opt/llvm/lib/libomp.dylib"
	HOMEBREW_LIBOMP="/opt/homebrew/opt/libomp/lib/libomp.dylib"
	for binary in \
		"$BUILD/bin/vnlbayes" \
		"$BUILD/bin/tvl1flow" \
		"$BUILD/lib/libvnlb.dylib" \
		"$BUILD/lib/libvidutils.dylib"
	do
		if [ -f "$binary" ]; then
			install_name_tool -change "$LLVM_LIBOMP" "$HOMEBREW_LIBOMP" "$binary"
		fi
	done
fi
