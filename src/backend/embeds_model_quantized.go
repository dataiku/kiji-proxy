//go:build embed && !trained
// +build embed,!trained

package main

import "embed"

// Embed the INT8-quantized model variant. This is the default for builds with
// `-tags=embed` and matches what the production build scripts (build_dmg.sh /
// build_linux.sh) stage into src/backend/model/quantized/. To embed the
// full-precision variant instead, build with `-tags=embed,trained`.
//
//go:embed model/quantized/*
var modelFiles embed.FS

// embeddedModelDir is the on-disk directory the embedded model files are
// extracted into at startup, and the directory the runtime loads from.
const embeddedModelDir = "model/quantized"
