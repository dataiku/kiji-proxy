package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// extractTempMarker is part of the name of every temp file written during
// extraction, so leftovers from an interrupted run can be found and removed.
const extractTempMarker = ".kiji-extract-"

// staleTempAge is how old a leftover temp file must be before it is removed.
// It is long enough that a concurrent extraction in another process is not
// disturbed.
const staleTempAge = time.Hour

// compareChunkSize is the read size used when comparing an embedded file
// with the copy already on disk.
const compareChunkSize = 1 << 20

// extractEmbeddedModelFiles extracts embedded model files into targetDir.
//
// Each file is streamed to disk (no full in-memory copy of model.onnx is
// kept in the Go heap) through a temp file that is renamed into place, so a
// failed or interrupted write never leaves a truncated file at the target
// path. Files already on disk with the same size and content are left
// untouched, so relaunches don't rewrite the model.
func extractEmbeddedModelFiles(modelFS fs.FS, targetDir string) error {
	if targetDir == "" {
		return fmt.Errorf("extractEmbeddedModelFiles: targetDir is empty (was the binary built with the embed tag?)")
	}
	if err := os.MkdirAll(targetDir, 0750); err != nil {
		return err
	}
	removeStaleExtractTemps(targetDir, staleTempAge)

	// Walk through embedded files and extract them
	return fs.WalkDir(modelFS, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		targetPath := filepath.Join(targetDir, filepath.Base(path))
		written, size, err := extractEmbeddedFile(modelFS, path, targetPath)
		if err != nil {
			return err
		}
		if written {
			log.Printf("Extracted: %s (size: %d bytes)", targetPath, size)
		} else {
			log.Printf("Up to date, not rewritten: %s (size: %d bytes)", targetPath, size)
		}
		return nil
	})
}

// extractEmbeddedFile writes the embedded file name to targetPath unless
// targetPath already holds the same bytes. It reports whether the file was
// written, and its size.
func extractEmbeddedFile(fsys fs.FS, name, targetPath string) (bool, int64, error) {
	info, err := fs.Stat(fsys, name)
	if err != nil {
		return false, 0, err
	}
	size := info.Size()

	same, err := sameContent(fsys, name, size, targetPath)
	if err != nil {
		return false, size, err
	}
	if same {
		return false, size, nil
	}
	if err := writeFileAtomic(fsys, name, targetPath); err != nil {
		return false, size, err
	}
	return true, size, nil
}

// sameContent reports whether targetPath is a regular file with the same
// size and bytes as the embedded file name. A missing or unreadable target
// counts as different, so the caller rewrites it.
func sameContent(fsys fs.FS, name string, size int64, targetPath string) (bool, error) {
	info, err := os.Stat(targetPath)
	if err != nil || !info.Mode().IsRegular() || info.Size() != size {
		return false, nil
	}

	src, err := fsys.Open(name)
	if err != nil {
		return false, err
	}
	defer func() { _ = src.Close() }()

	dst, err := os.Open(targetPath) // #nosec G304 — path is targetDir + embedded file name
	if err != nil {
		return false, nil
	}
	defer func() { _ = dst.Close() }()

	srcBuf := make([]byte, compareChunkSize)
	dstBuf := make([]byte, compareChunkSize)
	for {
		n, srcErr := io.ReadFull(src, srcBuf)
		if srcErr != nil && !errors.Is(srcErr, io.EOF) && !errors.Is(srcErr, io.ErrUnexpectedEOF) {
			return false, srcErr
		}
		m, dstErr := io.ReadFull(dst, dstBuf[:n])
		if dstErr != nil && !errors.Is(dstErr, io.EOF) && !errors.Is(dstErr, io.ErrUnexpectedEOF) {
			return false, nil
		}
		if m != n || !bytes.Equal(srcBuf[:n], dstBuf[:m]) {
			return false, nil
		}
		if srcErr != nil {
			// Source exhausted; sizes were equal, so the target is too.
			return true, nil
		}
	}
}

// writeFileAtomic streams the embedded file name into a temp file next to
// targetPath and renames it into place. On any error the temp file is
// removed and targetPath is left as it was.
func writeFileAtomic(fsys fs.FS, name, targetPath string) (err error) {
	src, err := fsys.Open(name)
	if err != nil {
		return err
	}
	defer func() { _ = src.Close() }()

	dir := filepath.Dir(targetPath)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(targetPath)+extractTempMarker+"*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		if err != nil {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	if _, err = io.Copy(tmp, src); err != nil {
		return fmt.Errorf("write %s: %w", targetPath, err)
	}
	if err = tmp.Chmod(0600); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, targetPath)
}

// removeStaleExtractTemps deletes temp files left in dir by an extraction
// that was interrupted (for example by a crash) at least maxAge ago.
func removeStaleExtractTemps(dir string, maxAge time.Duration) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-maxAge)
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), ".") || !strings.Contains(e.Name(), extractTempMarker) {
			continue
		}
		info, err := e.Info()
		if err != nil || info.ModTime().After(cutoff) {
			continue
		}
		path := filepath.Join(dir, e.Name())
		if err := os.Remove(path); err == nil {
			log.Printf("Removed leftover extraction temp file: %s", path)
		}
	}
}
