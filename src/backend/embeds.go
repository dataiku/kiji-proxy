//go:build embed
// +build embed

package main

import "embed"

// Embed frontend UI files. All embed builds ship the web UI; whether it is
// actually served on "/" is gated at runtime by config.ServeUI (KIJI_SERVE_UI).
//
//go:embed frontend/dist/*
var uiFiles embed.FS
