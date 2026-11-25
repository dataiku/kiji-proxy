#!/bin/bash

# Build a .deb for the Linux server build using debhelper / dpkg-buildpackage.
# Consumes the staging tree produced by src/scripts/build_linux.sh; if only
# the tarball is present, it is re-extracted in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PKG="kiji-privacy-proxy"
RELEASE_DIR="release/linux"
DEBIAN_SRC="packaging/debian"

VERSION=$(cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "0.0.0")

echo "🔨 Building ${PKG} .deb (version ${VERSION})"
echo "============================================"

for tool in dpkg-buildpackage dh fakeroot; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "❌ Missing required tool: $tool" >&2
        echo "   Install with: sudo apt-get install build-essential debhelper devscripts fakeroot" >&2
        exit 1
    fi
done

STAGED_NAME="${PKG}-${VERSION}-linux-amd64"
STAGED_DIR="${RELEASE_DIR}/${STAGED_NAME}"
TARBALL="${RELEASE_DIR}/${STAGED_NAME}.tar.gz"

if [ ! -d "$STAGED_DIR" ]; then
    if [ ! -f "$TARBALL" ]; then
        echo "❌ Neither staging dir nor tarball found:" >&2
        echo "   - $STAGED_DIR" >&2
        echo "   - $TARBALL" >&2
        echo "   Run src/scripts/build_linux.sh first." >&2
        exit 1
    fi
    echo "📦 Re-extracting staging tree from $(basename "$TARBALL")"
    tar -xzf "$TARBALL" -C "$RELEASE_DIR"
fi

if [ -e debian ]; then
    echo "❌ A 'debian' entry already exists at the project root; aborting to avoid clobbering it." >&2
    exit 1
fi

cp -a "$DEBIAN_SRC" debian
chmod +x debian/rules debian/kiji-proxy.wrapper

cleanup() {
    rm -rf debian
}
trap cleanup EXIT

DATE=$(date -R)
cat > debian/changelog <<EOF
${PKG} (${VERSION}) unstable; urgency=medium

  * Release ${VERSION}. See CHANGELOG.md for details.

 -- Dataiku Open Source Lab <opensource@dataiku.com>  ${DATE}
EOF

echo "📦 Running dpkg-buildpackage..."
dpkg-buildpackage -b -us -uc

PARENT="$(cd .. && pwd)"
DEB_BASENAME="${PKG}_${VERSION}_amd64.deb"
DEB_FILE="${PARENT}/${DEB_BASENAME}"

if [ ! -f "$DEB_FILE" ]; then
    echo "❌ Expected .deb not found at $DEB_FILE" >&2
    ls "$PARENT" | grep -E "^${PKG}_${VERSION}" || true
    exit 1
fi

mv "$DEB_FILE" "${RELEASE_DIR}/"
mv "${PARENT}/${PKG}_${VERSION}_amd64.buildinfo" "${RELEASE_DIR}/" 2>/dev/null || true
mv "${PARENT}/${PKG}_${VERSION}_amd64.changes"   "${RELEASE_DIR}/" 2>/dev/null || true

cd "$RELEASE_DIR"
sha256sum "$DEB_BASENAME" > "${DEB_BASENAME}.sha256"
cd "$PROJECT_ROOT"

echo ""
echo "✅ Deb package built:"
echo "   ${RELEASE_DIR}/${DEB_BASENAME}"
echo "   $(cat "${RELEASE_DIR}/${DEB_BASENAME}.sha256")"
