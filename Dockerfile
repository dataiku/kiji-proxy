# Build stage
FROM golang:1.21-alpine AS builder

# Set working directory
WORKDIR /app

# Install git and ca-certificates (needed for go mod download)
RUN apk add --no-cache git ca-certificates

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code - exclude frontend/, model-server/, and pii_model/
COPY src/backend/main.go ./src/backend/
COPY src/backend/config/ ./src/backend/config/
COPY src/backend/pii/ ./src/backend/pii/
COPY src/backend/processor/ ./src/backend/processor/
COPY src/backend/proxy/ ./src/backend/proxy/
COPY src/backend/server/ ./src/backend/server/
COPY pii_onnx_model/ ./pii_onnx_model/
COPY build/tokenizers/ ./build/tokenizers/

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./src/backend

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/main .

# Copy scripts directory (for reference)
COPY --from=builder /app/scripts ./scripts

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./main"]
