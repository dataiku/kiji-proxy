//go:build embed
// +build embed

package main

import "embed"

// Embed frontend UI files
//
//go:embed frontend/dist/*
var uiFiles embed.FS

// Embed model files
//
//go:embed model/quantized/*
var modelFiles embed.FS
