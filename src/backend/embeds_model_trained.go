//go:build embed && trained
// +build embed,trained

package main

import "embed"

// Embed the full-precision (trained) model variant. Selected by adding
// `-tags=embed,trained` at build time. The build pipeline must stage the model
// files into src/backend/model/trained/ before invoking `go build` (the
// production build_dmg.sh / build_linux.sh scripts only stage the quantized
// variant today).
//
//go:embed model/trained/*
var modelFiles embed.FS

// embeddedModelDir is the on-disk directory the embedded model files are
// extracted into at startup, and the directory the runtime loads from.
const embeddedModelDir = "model/trained"
