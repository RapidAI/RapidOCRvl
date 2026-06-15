#!/bin/sh
set -e

LABEL="com.znsoft.paddleocrvl.service"
PLIST="/Library/LaunchDaemons/$LABEL.plist"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root: sudo /usr/local/paddleocrvl/uninstall.sh" >&2
  exit 1
fi

if [ -f "$PLIST" ]; then
  /bin/launchctl bootout system "$PLIST" >/dev/null 2>&1 || true
fi

/bin/rm -f "$PLIST"
/bin/rm -rf "/Applications/PaddleOCR-VL Client.app"
/bin/rm -rf "/usr/local/paddleocrvl"

echo "PaddleOCR-VL service and application removed."
echo "Model/admin data kept at /Library/Application Support/PaddleOCRVL"
