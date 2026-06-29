package pii

import (
	"os"
	"path/filepath"
	"testing"
)

// writeManifest writes model_manifest.json with the given contents into a fresh
// temp dir and returns the dir path.
func writeManifest(t *testing.T, contents string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model_manifest.json"), []byte(contents), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return dir
}

func TestReadModelIdentity(t *testing.T) {
	tests := []struct {
		name     string
		manifest string // when empty, no manifest file is written
		wantSig  string
		wantHash string
	}{
		{
			name:     "missing manifest falls back to default signature",
			manifest: "",
			wantSig:  "onnx-pii",
			wantHash: "",
		},
		{
			name:     "malformed json falls back to default signature",
			manifest: "{ not json",
			wantSig:  "onnx-pii",
			wantHash: "",
		},
		{
			name:     "model field wins as signature",
			manifest: `{"model":"distilbert-pii","name":"ignored","version":"9"}`,
			wantSig:  "distilbert-pii",
			wantHash: "",
		},
		{
			name:     "name used when model absent",
			manifest: `{"name":"pii-ner","version":"9"}`,
			wantSig:  "pii-ner",
			wantHash: "",
		},
		{
			name:     "version used when model and name absent",
			manifest: `{"version":"2026.06"}`,
			wantSig:  "2026.06",
			wantHash: "",
		},
		{
			name:     "sha256 is truncated to 7 chars",
			manifest: `{"model":"m","hashes":{"sha256":"2b657df1f9a2deadbeef"}}`,
			wantSig:  "m",
			wantHash: "2b657df",
		},
		{
			name:     "no recognized signature key but a hash (mirrors the quantized model)",
			manifest: `{"hashes":{"sha256":"2b657df1f9a2"}}`,
			wantSig:  "onnx-pii",
			wantHash: "2b657df",
		},
		{
			name:     "short sha256 is ignored",
			manifest: `{"model":"m","hashes":{"sha256":"abc"}}`,
			wantSig:  "m",
			wantHash: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var dir string
			if tc.manifest == "" {
				dir = t.TempDir() // exists, but contains no manifest
			} else {
				dir = writeManifest(t, tc.manifest)
			}
			sig, hash := readModelIdentity(dir)
			if sig != tc.wantSig {
				t.Errorf("signature = %q, want %q", sig, tc.wantSig)
			}
			if hash != tc.wantHash {
				t.Errorf("hash = %q, want %q", hash, tc.wantHash)
			}
		})
	}
}

// TestReadModelIdentity_TracksDirectory is the regression guard for the bug fix:
// identity is derived from the directory passed in, so two different model
// directories yield their own signature/hash rather than a single cached value.
func TestReadModelIdentity_TracksDirectory(t *testing.T) {
	dirA := writeManifest(t, `{"model":"model-a","hashes":{"sha256":"aaaaaaa0000"}}`)
	dirB := writeManifest(t, `{"model":"model-b","hashes":{"sha256":"bbbbbbb1111"}}`)

	sigA, hashA := readModelIdentity(dirA)
	sigB, hashB := readModelIdentity(dirB)

	if sigA != "model-a" || hashA != "aaaaaaa" {
		t.Errorf("dir A identity = (%q,%q), want (model-a, aaaaaaa)", sigA, hashA)
	}
	if sigB != "model-b" || hashB != "bbbbbbb" {
		t.Errorf("dir B identity = (%q,%q), want (model-b, bbbbbbb)", sigB, hashB)
	}
}

// TestGetModelIdentity confirms the accessor returns the fields set at load time.
func TestGetModelIdentity(t *testing.T) {
	mm := &ModelManager{modelSignature: "sig-x", modelHash: "hash-y"}
	sig, hash := mm.GetModelIdentity()
	if sig != "sig-x" || hash != "hash-y" {
		t.Errorf("GetModelIdentity() = (%q,%q), want (sig-x, hash-y)", sig, hash)
	}
}
