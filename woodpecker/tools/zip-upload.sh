SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
rm -f upload.zip
zip -r upload data
"$SCRIPT_DIR/upload-to-bee.sh" upload.zip
