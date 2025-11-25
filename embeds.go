//go:build embed
// +build embed

package main

import "embed"

//go:embed electron_ui/dist/*
var uiFiles embed.FS

//go:embed pii_onnx_model/*
var modelFiles embed.FS
