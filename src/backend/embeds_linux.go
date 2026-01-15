//go:build embed && linux
// +build embed,linux

package main

import "embed"

// Linux builds are API-only, so we don't embed the UI files
var uiFiles embed.FS

// Embed model files
//
//go:embed model/quantized/*
var modelFiles embed.FS
