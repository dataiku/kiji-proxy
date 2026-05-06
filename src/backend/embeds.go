//go:build embed && !linux
// +build embed,!linux

package main

import "embed"

// Embed frontend UI files (macOS/Windows builds with Electron)
//
//go:embed frontend/dist/*
var uiFiles embed.FS
