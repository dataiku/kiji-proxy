//go:build embed
// +build embed

package main

import "embed"

//go:embed frontend/dist/*
var uiFiles embed.FS

//go:embed model/quantized/*
var modelFiles embed.FS
