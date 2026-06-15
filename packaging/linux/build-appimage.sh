#!/bin/sh
set -e

VERSION="${VERSION:-1.0.0}"
ARCH="${ARCH:-$(uname -m)}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
DIST="$REPO/dist/linux"
APPDIR="$DIST/PaddleOCR-VL.AppDir"

case "$ARCH" in
  x86_64|amd64)
    APPIMAGE_ARCH="x86_64"
    GOARCH="amd64"
    ;;
  aarch64|arm64)
    APPIMAGE_ARCH="aarch64"
    GOARCH="arm64"
    ;;
  *)
    echo "Unsupported ARCH: $ARCH" >&2
    exit 1
    ;;
esac

SERVER="$APPDIR/usr/bin/paddleocrvl-server"
CLIENT_SRC="$REPO/cmd/paddleocrvl-client/build/bin/paddleocrvl-client"
CLIENT_DST="$APPDIR/usr/bin/paddleocrvl-client"
LINUXDEPLOY="$DIST/linuxdeploy-$APPIMAGE_ARCH.AppImage"
GTK_PLUGIN="$DIST/linuxdeploy-plugin-gtk.sh"
OUT="$DIST/PaddleOCR-VL-$VERSION-linux-$APPIMAGE_ARCH.AppImage"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$DIST"

cd "$REPO"
GOOS=linux GOARCH="$GOARCH" go build -trimpath -ldflags "-s -w" -o "$SERVER" ./cmd/paddleocrvl-server

cd "$REPO/cmd/paddleocrvl-client"
wails build -platform "linux/$GOARCH" -clean
if [ ! -f "$CLIENT_SRC" ]; then
  echo "Wails client binary not found: $CLIENT_SRC" >&2
  exit 1
fi
cp "$CLIENT_SRC" "$CLIENT_DST"
chmod 755 "$SERVER" "$CLIENT_DST"

cp "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
cp "$SCRIPT_DIR/paddleocrvl-client.desktop" "$APPDIR/paddleocrvl-client.desktop"
cp "$SCRIPT_DIR/paddleocrvl-client.desktop" "$APPDIR/usr/share/applications/paddleocrvl-client.desktop"
cp "$REPO/cmd/paddleocrvl-client/build/appicon.png" "$APPDIR/paddleocrvl-client.png"
cp "$REPO/cmd/paddleocrvl-client/build/appicon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/paddleocrvl-client.png"
chmod 755 "$APPDIR/AppRun"

if [ ! -x "$LINUXDEPLOY" ]; then
  URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-$APPIMAGE_ARCH.AppImage"
  curl -L "$URL" -o "$LINUXDEPLOY"
  chmod 755 "$LINUXDEPLOY"
fi
if [ ! -x "$GTK_PLUGIN" ]; then
  curl -L "https://raw.githubusercontent.com/linuxdeploy/linuxdeploy-plugin-gtk/master/linuxdeploy-plugin-gtk.sh" -o "$GTK_PLUGIN"
  chmod 755 "$GTK_PLUGIN"
fi

cd "$DIST"
rm -f "$OUT"
rm -f "$DIST"/PaddleOCR-VL*.AppImage "$DIST"/*.AppImage.tmp
PATH="$DIST:$PATH" \
APPIMAGE_EXTRACT_AND_RUN=1 \
ARCH="$APPIMAGE_ARCH" \
DEPLOY_GTK_VERSION=3 \
LDAI_OUTPUT="$OUT" \
LINUXDEPLOY_OUTPUT_VERSION="$VERSION" \
"$LINUXDEPLOY" \
  --appdir "$APPDIR" \
  --desktop-file "$APPDIR/paddleocrvl-client.desktop" \
  --icon-file "$APPDIR/paddleocrvl-client.png" \
  --executable "$CLIENT_DST" \
  --executable "$SERVER" \
  --plugin gtk \
  --output appimage
if [ ! -f "$OUT" ]; then
  GENERATED="$(find "$DIST" -maxdepth 1 -type f -name "*.AppImage" ! -name "linuxdeploy-*.AppImage" | sort | tail -n 1)"
  if [ -z "$GENERATED" ]; then
    echo "linuxdeploy did not create an AppImage" >&2
    exit 1
  fi
  mv "$GENERATED" "$OUT"
fi
chmod 755 "$OUT"
echo "Created $OUT"
