//go:build embed
// +build embed

package main

import "embed"

// Frontend embedding is disabled for Linux builds (API-only)
// Only the model files are embedded
var uiFiles embed.FS

//go:embed model/quantized/*
var modelFiles embed.FS
