FROM --platform=linux/amd64 ubuntu:24.04

ARG KIJI_VERSION=""

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Download and install the latest (or specified) Linux release from GitHub
RUN set -eux; \
    REPO="dataiku/kiji-proxy"; \
    if [ -n "$KIJI_VERSION" ]; then \
        TAG="v${KIJI_VERSION}"; \
    else \
        TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | jq -r '.tag_name'); \
        KIJI_VERSION="${TAG#v}"; \
    fi; \
    ARCHIVE="kiji-privacy-proxy-${KIJI_VERSION}-linux-amd64.tar.gz"; \
    URL="https://github.com/${REPO}/releases/download/${TAG}/${ARCHIVE}"; \
    echo "Downloading ${URL}"; \
    curl -fsSL -o "/tmp/${ARCHIVE}" "${URL}"; \
    # Verify checksum
    curl -fsSL -o "/tmp/${ARCHIVE}.sha256" "${URL}.sha256"; \
    cd /tmp && sha256sum -c "${ARCHIVE}.sha256"; \
    # Extract to /opt
    mkdir -p /opt/kiji-proxy; \
    tar -xzf "/tmp/${ARCHIVE}" -C /opt/kiji-proxy --strip-components=1; \
    rm -f "/tmp/${ARCHIVE}" "/tmp/${ARCHIVE}.sha256"; \
    chmod +x /opt/kiji-proxy/bin/kiji-proxy /opt/kiji-proxy/run.sh

ENV LD_LIBRARY_PATH="/opt/kiji-proxy/lib"
ENV ONNXRUNTIME_SHARED_LIBRARY_PATH="/opt/kiji-proxy/lib/libonnxruntime.so.1.24.2"

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Copy and run the test script
COPY docker-test-api.sh /opt/kiji-proxy/docker-test-api.sh
RUN chmod +x /opt/kiji-proxy/docker-test-api.sh

WORKDIR /opt/kiji-proxy

# Default command runs the proxy; override with the test script path to run tests
CMD ["/opt/kiji-proxy/bin/kiji-proxy"]
