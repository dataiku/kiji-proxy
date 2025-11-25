//go:build !embed
// +build !embed

package main

import "embed"

// Stub embed.FS variables for linting/development when embed tag is not set
// These will be empty but allow the code to compile without requiring the files
var uiFiles embed.FS
var modelFiles embed.FS
