//go:build !embed
// +build !embed

package main

import "embed"

// Stub embed.FS variables for linting/development when embed tag is not set.
// These are empty so the code compiles without the model/UI files. In dev mode
// the runtime loads model files from the filesystem path returned by
// cfg.ResolveModelDirectory().
var uiFiles embed.FS
var modelFiles embed.FS

// embeddedModelDir is empty when nothing is embedded. main.go falls back to
// cfg.ResolveModelDirectory() in this mode.
const embeddedModelDir = ""
