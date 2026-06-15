#!/bin/sh
set -e

VERSION="${VERSION:-1.0.0}"
PKG_VERSION="$(printf "%s" "$VERSION" | sed -E 's/^[vV]//; s/[-+].*$//; s/[^0-9.].*$//')"
if [ -z "$PKG_VERSION" ]; then
  PKG_VERSION="1.0.0"
fi
IDENTIFIER="${IDENTIFIER:-com.znsoft.paddleocrvl}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
DIST="$REPO/dist/macos"
WORK="$DIST/pkgroot"
SERVER_AMD64="$DIST/paddleocrvl-server-amd64"
SERVER_ARM64="$DIST/paddleocrvl-server-arm64"
SERVER_UNIVERSAL="$WORK/usr/local/paddleocrvl/paddleocrvl-server"
APP_DST="$WORK/Applications/PaddleOCR-VL Client.app"
COMPONENT="$DIST/PaddleOCR-VL-$VERSION-component.pkg"
PKG="$DIST/PaddleOCR-VL-$VERSION-macos-universal.pkg"

rm -rf "$WORK"
mkdir -p "$WORK/usr/local/paddleocrvl" \
  "$WORK/Applications" \
  "$WORK/Library/Application Support/PaddleOCRVL/models" \
  "$WORK/Library/LaunchDaemons" \
  "$DIST"

cd "$REPO"
GOOS=darwin GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o "$SERVER_AMD64" ./cmd/paddleocrvl-server
GOOS=darwin GOARCH=arm64 go build -trimpath -ldflags "-s -w" -o "$SERVER_ARM64" ./cmd/paddleocrvl-server
lipo -create -output "$SERVER_UNIVERSAL" "$SERVER_AMD64" "$SERVER_ARM64"
chmod 755 "$SERVER_UNIVERSAL"
cp "$SCRIPT_DIR/uninstall.sh" "$WORK/usr/local/paddleocrvl/uninstall.sh"
chmod 755 "$WORK/usr/local/paddleocrvl/uninstall.sh"

cd "$REPO/cmd/paddleocrvl-client"
wails build -platform darwin/universal -clean
APP_SRC="$REPO/cmd/paddleocrvl-client/build/bin/PaddleOCR-VL Client.app"
if [ ! -d "$APP_SRC" ]; then
  APP_SRC="$REPO/cmd/paddleocrvl-client/build/bin/paddleocrvl-client.app"
fi
if [ ! -d "$APP_SRC" ]; then
  echo "Wails app not found under cmd/paddleocrvl-client/build/bin" >&2
  exit 1
fi
cp -R "$APP_SRC" "$APP_DST"

cp "$SCRIPT_DIR/com.znsoft.paddleocrvl.service.plist" "$WORK/Library/LaunchDaemons/com.znsoft.paddleocrvl.service.plist"
chmod 644 "$WORK/Library/LaunchDaemons/com.znsoft.paddleocrvl.service.plist"
chmod +x "$SCRIPT_DIR/scripts/preinstall" "$SCRIPT_DIR/scripts/postinstall"

pkgbuild \
  --root "$WORK" \
  --scripts "$SCRIPT_DIR/scripts" \
  --identifier "$IDENTIFIER" \
  --version "$PKG_VERSION" \
  --install-location "/" \
  "$COMPONENT"

productbuild \
  --package "$COMPONENT" \
  "$PKG"

echo "Created $PKG"
