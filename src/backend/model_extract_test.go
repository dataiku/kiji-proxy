package main

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

func testModelFS(model []byte) fstest.MapFS {
	return fstest.MapFS{
		"model/quantized/model.onnx":     {Data: model},
		"model/quantized/tokenizer.json": {Data: []byte(`{"tok":1}`)},
		"model/quantized/sub/vocab.txt":  {Data: []byte("a\nb\n")},
	}
}

// bigModel spans several compare chunks with a tail, so chunked comparison
// is exercised across boundaries.
func bigModel() []byte {
	b := make([]byte, 2*compareChunkSize+123)
	for i := range b {
		b[i] = byte(i * 31)
	}
	return b
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path) // #nosec G304 — test temp dir
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func assertNoTempFiles(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.Contains(e.Name(), extractTempMarker) {
			t.Fatalf("temp file left behind: %s", e.Name())
		}
	}
}

func TestExtractEmbeddedModelFilesFresh(t *testing.T) {
	model := bigModel()
	dir := filepath.Join(t.TempDir(), "model", "quantized")

	if err := extractEmbeddedModelFiles(testModelFS(model), dir); err != nil {
		t.Fatal(err)
	}

	if got := readFile(t, filepath.Join(dir, "model.onnx")); !bytes.Equal(got, model) {
		t.Fatal("model.onnx content differs from embedded file")
	}
	if got := string(readFile(t, filepath.Join(dir, "tokenizer.json"))); got != `{"tok":1}` {
		t.Fatalf("tokenizer.json = %q", got)
	}
	// Nested embedded paths are flattened into targetDir, as before.
	if got := string(readFile(t, filepath.Join(dir, "vocab.txt"))); got != "a\nb\n" {
		t.Fatalf("vocab.txt = %q", got)
	}
	info, err := os.Stat(filepath.Join(dir, "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0600 {
		t.Fatalf("model.onnx mode = %o, want 600", perm)
	}
	assertNoTempFiles(t, dir)
}

func TestExtractEmbeddedModelFilesSkipsUnchanged(t *testing.T) {
	model := bigModel()
	dir := t.TempDir()
	modelFS := testModelFS(model)

	if err := extractEmbeddedModelFiles(modelFS, dir); err != nil {
		t.Fatal(err)
	}
	// Backdate the files: a rewrite would give them a fresh mtime.
	old := time.Now().Add(-48 * time.Hour).Truncate(time.Second)
	for _, name := range []string{"model.onnx", "tokenizer.json", "vocab.txt"} {
		if err := os.Chtimes(filepath.Join(dir, name), old, old); err != nil {
			t.Fatal(err)
		}
	}

	written, _, err := extractEmbeddedFile(modelFS, "model/quantized/model.onnx", filepath.Join(dir, "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	if written {
		t.Fatal("unchanged model.onnx was rewritten")
	}
	if err := extractEmbeddedModelFiles(modelFS, dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"model.onnx", "tokenizer.json", "vocab.txt"} {
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if !info.ModTime().Equal(old) {
			t.Fatalf("%s was rewritten (mtime %v, want %v)", name, info.ModTime(), old)
		}
	}
}

func TestExtractEmbeddedModelFilesRewritesChangedContent(t *testing.T) {
	model := bigModel()
	dir := t.TempDir()
	target := filepath.Join(dir, "model.onnx")

	cases := map[string][]byte{
		// Same size, one byte flipped in the last, partial chunk.
		"same size": func() []byte {
			b := bytes.Clone(model)
			b[len(b)-1] ^= 0xff
			return b
		}(),
		"shorter": model[:len(model)-1],
		"longer":  append(bytes.Clone(model), 0),
	}
	for name, stale := range cases {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(target, stale, 0600); err != nil {
				t.Fatal(err)
			}
			if err := extractEmbeddedModelFiles(testModelFS(model), dir); err != nil {
				t.Fatal(err)
			}
			if got := readFile(t, target); !bytes.Equal(got, model) {
				t.Fatal("stale model.onnx was not replaced")
			}
		})
	}
}

// failingFS serves files from a MapFS, but reading failName fails after
// the first few bytes.
type failingFS struct {
	fstest.MapFS
	failName string
}

var errInjected = errors.New("injected read failure")

func (f failingFS) Open(name string) (fs.File, error) {
	file, err := f.MapFS.Open(name)
	if err != nil || name != f.failName {
		return file, err
	}
	return &failingFile{File: file}, nil
}

type failingFile struct {
	fs.File
	read int
}

func (f *failingFile) Read(p []byte) (int, error) {
	if f.read >= 4 {
		return 0, errInjected
	}
	if len(p) > 4-f.read {
		p = p[:4-f.read]
	}
	n, err := f.File.Read(p)
	f.read += n
	return n, err
}

func TestExtractEmbeddedModelFilesAtomicOnFailure(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "model.onnx")
	previous := []byte("previous model, different size")
	if err := os.WriteFile(target, previous, 0600); err != nil {
		t.Fatal(err)
	}

	modelFS := failingFS{MapFS: testModelFS(bigModel()), failName: "model/quantized/model.onnx"}
	err := extractEmbeddedModelFiles(modelFS, dir)
	if !errors.Is(err, errInjected) {
		t.Fatalf("err = %v, want injected failure", err)
	}
	if got := readFile(t, target); !bytes.Equal(got, previous) {
		t.Fatal("failed extraction modified the existing model.onnx")
	}
	assertNoTempFiles(t, dir)

	// Fresh directory: a failed write leaves no partial target behind.
	fresh := t.TempDir()
	if err := extractEmbeddedModelFiles(modelFS, fresh); !errors.Is(err, errInjected) {
		t.Fatalf("err = %v, want injected failure", err)
	}
	if _, err := os.Stat(filepath.Join(fresh, "model.onnx")); !os.IsNotExist(err) {
		t.Fatalf("partial model.onnx exists after failed extraction (stat err %v)", err)
	}
	assertNoTempFiles(t, fresh)
}

func TestExtractEmbeddedModelFilesRemovesStaleTemps(t *testing.T) {
	dir := t.TempDir()
	stale := filepath.Join(dir, ".model.onnx"+extractTempMarker+"123")
	recent := filepath.Join(dir, ".model.onnx"+extractTempMarker+"456")
	for _, p := range []string{stale, recent} {
		if err := os.WriteFile(p, []byte("partial"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	old := time.Now().Add(-2 * staleTempAge)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatal(err)
	}

	if err := extractEmbeddedModelFiles(testModelFS([]byte("m")), dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatal("stale temp file was not removed")
	}
	// A recent temp file may belong to a concurrent extraction; keep it.
	if _, err := os.Stat(recent); err != nil {
		t.Fatalf("recent temp file was removed: %v", err)
	}
}

func TestExtractEmbeddedModelFilesEmptyTargetDir(t *testing.T) {
	if err := extractEmbeddedModelFiles(testModelFS(nil), ""); err == nil {
		t.Fatal("expected an error for an empty targetDir")
	}
}
