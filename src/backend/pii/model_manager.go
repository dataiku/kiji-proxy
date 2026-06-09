package pii

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	pii "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

// ModelManager manages PII model lifecycle with thread-safe hot reload capability.
// Besides the hot-reloadable ONNX detector it also holds any static extra
// detectors (e.g. the regex detector). GetDetector hands out all enabled
// detectors combined into a single CompositeDetector, so callers stay unaware of
// how many detectors are running.
type ModelManager struct {
	mu              sync.RWMutex
	currentDetector pii.Detector
	// extraDetectors are non-ONNX detectors (e.g. regex) supplied at construction.
	// They are static across model hot-reloads and are only closed on Close().
	extraDetectors []pii.Detector
	// onnxEnabled controls whether the ONNX detector participates in detection.
	// When false, only the extra detectors are used (the model may still be loaded
	// so the health/reload endpoints keep working).
	onnxEnabled               bool
	modelDirectory            string
	isHealthy                 bool
	lastError                 error
	entityConfidenceThreshold float64
}

// ModelConfig holds paths to required model files
type ModelConfig struct {
	ModelPath     string
	TokenizerPath string
	LabelMapPath  string
}

// NewModelManager creates a new model manager and initializes with the given
// directory. onnxEnabled controls whether the ONNX model participates in
// detection; extraDetectors are static non-ONNX detectors (e.g. regex) that run
// alongside it and survive model hot-reloads.
func NewModelManager(directory string, onnxEnabled bool, extraDetectors ...pii.Detector) (*ModelManager, error) {
	mm := &ModelManager{
		modelDirectory:            directory,
		onnxEnabled:               onnxEnabled,
		extraDetectors:            extraDetectors,
		isHealthy:                 false,
		entityConfidenceThreshold: 0.25,
	}

	// Apply the initial threshold to the static detectors. The ONNX detector gets
	// the threshold applied during ReloadModel below.
	for _, d := range mm.extraDetectors {
		d.SetEntityConfidenceThreshold(mm.entityConfidenceThreshold)
	}

	// Perform initial load - don't fail if model can't load, just mark as unhealthy
	if err := mm.ReloadModel(directory); err != nil {
		log.Printf("[ModelManager] Warning: Failed to load initial model: %v", err)
		log.Printf("[ModelManager] Model manager created but marked as unhealthy")
		// Don't return error - allow server to start with unhealthy model
		// This matches the behavior before ModelManager was introduced
	}

	return mm, nil
}

// GetDetector returns the active detector in a thread-safe manner. The returned
// detector combines the ONNX model (when enabled and healthy) with any extra
// detectors. With a single active detector it is returned directly; with several
// it is wrapped in a CompositeDetector. If the ONNX model is unhealthy but extra
// detectors exist, detection degrades to those rather than failing entirely.
func (mm *ModelManager) GetDetector() (pii.Detector, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	var active []pii.Detector
	if mm.onnxEnabled {
		if mm.isHealthy && mm.currentDetector != nil {
			active = append(active, mm.currentDetector)
		} else if len(mm.extraDetectors) == 0 {
			// ONNX is the only configured detector and it is unavailable.
			return nil, fmt.Errorf("model is unhealthy: %w", mm.lastError)
		}
	}
	active = append(active, mm.extraDetectors...)

	switch len(active) {
	case 0:
		return nil, fmt.Errorf("no detectors available")
	case 1:
		return active[0], nil
	default:
		return pii.NewCompositeDetector(active...), nil
	}
}

// ReloadModel reloads the model from the specified directory with validation
func (mm *ModelManager) ReloadModel(newDirectory string) error {
	log.Printf("[ModelManager] Reloading model from directory: %s", newDirectory)

	// Step 1: Validate directory structure
	config, err := mm.validateDirectory(newDirectory)
	if err != nil {
		mm.mu.Lock()
		mm.isHealthy = false
		mm.lastError = err
		mm.mu.Unlock()
		log.Printf("[ModelManager] Directory validation failed: %v", err)
		return fmt.Errorf("validation failed: %w", err)
	}

	// Step 2: Attempt to load new detector (outside lock to minimize blocking)
	log.Printf("[ModelManager] Loading new detector from: %s", config.ModelPath)
	newDetector, err := pii.NewONNXModelDetectorSimple(
		config.ModelPath,
		config.TokenizerPath,
	)
	if err != nil {
		mm.mu.Lock()
		mm.isHealthy = false
		mm.lastError = err
		mm.mu.Unlock()
		log.Printf("[ModelManager] Failed to load model: %v", err)
		return fmt.Errorf("failed to load model: %w", err)
	}

	// Step 3: Run validation inference to ensure model works
	log.Printf("[ModelManager] Running validation inference")
	testInput := pii.DetectorInput{Text: "Test with John Smith"}
	_, err = newDetector.Detect(context.Background(), testInput)
	if err != nil {
		// Close the failed detector
		if closeErr := newDetector.Close(); closeErr != nil {
			log.Printf("[ModelManager] Warning: failed to close failed detector: %v", closeErr)
		}

		mm.mu.Lock()
		mm.isHealthy = false
		mm.lastError = err
		mm.mu.Unlock()
		log.Printf("[ModelManager] Model validation inference failed: %v", err)
		return fmt.Errorf("model validation failed: %w", err)
	}

	// Step 4: Apply stored confidence threshold and swap detectors atomically
	mm.mu.Lock()
	newDetector.SetEntityConfidenceThreshold(mm.entityConfidenceThreshold)
	oldDetector := mm.currentDetector
	mm.currentDetector = newDetector
	mm.modelDirectory = newDirectory
	mm.isHealthy = true
	mm.lastError = nil
	mm.mu.Unlock()

	log.Printf("[ModelManager] Model swap completed successfully")

	// Step 5: Close old detector outside lock to minimize critical section
	if oldDetector != nil {
		log.Printf("[ModelManager] Closing old detector")
		if err := oldDetector.Close(); err != nil {
			log.Printf("[ModelManager] Warning: failed to close old detector: %v", err)
		}
	}

	log.Printf("[ModelManager] Model reload complete for directory: %s", newDirectory)
	return nil
}

// IsHealthy returns whether the current model is healthy
func (mm *ModelManager) IsHealthy() bool {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.isHealthy
}

// GetLastError returns the last error encountered (if any)
func (mm *ModelManager) GetLastError() error {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.lastError
}

// SetEntityConfidenceThreshold updates the confidence threshold on the model
// manager and every active detector (ONNX plus the extra detectors).
func (mm *ModelManager) SetEntityConfidenceThreshold(threshold float64) {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	mm.entityConfidenceThreshold = threshold
	if mm.currentDetector != nil {
		mm.currentDetector.SetEntityConfidenceThreshold(threshold)
	}
	for _, d := range mm.extraDetectors {
		d.SetEntityConfidenceThreshold(threshold)
	}
}

// GetEntityConfidenceThreshold returns the current confidence threshold
func (mm *ModelManager) GetEntityConfidenceThreshold() float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.entityConfidenceThreshold
}

// activeDetectorNamesLocked returns the names of the detectors GetDetector would
// currently hand out, in the same order: the ONNX detector (only when enabled and
// healthy) followed by the extra detectors. Callers must hold mm.mu.
func (mm *ModelManager) activeDetectorNamesLocked() []string {
	names := make([]string, 0, len(mm.extraDetectors)+1)
	if mm.onnxEnabled && mm.isHealthy && mm.currentDetector != nil {
		names = append(names, mm.currentDetector.GetName())
	}
	for _, d := range mm.extraDetectors {
		names = append(names, d.GetName())
	}
	return names
}

// GetInfo returns information about the current model state
func (mm *ModelManager) GetInfo() map[string]interface{} {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	info := map[string]interface{}{
		"directory":    mm.modelDirectory,
		"healthy":      mm.isHealthy,
		"confidence":   mm.entityConfidenceThreshold,
		"onnx_enabled": mm.onnxEnabled,
		"detectors":    mm.activeDetectorNamesLocked(),
	}

	if mm.lastError != nil {
		info["error"] = mm.lastError.Error()
	} else {
		info["error"] = nil
	}

	return info
}

// validateDirectory checks that the directory exists and contains all required files
func (mm *ModelManager) validateDirectory(dir string) (*ModelConfig, error) {
	// Check directory exists
	info, err := os.Stat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("directory does not exist: %s", dir)
		}
		return nil, fmt.Errorf("failed to access directory: %w", err)
	}

	if !info.IsDir() {
		return nil, fmt.Errorf("path is not a directory: %s", dir)
	}

	modelFilename := "model.onnx"
	if _, err := os.Stat(filepath.Join(dir, modelFilename)); os.IsNotExist(err) {
		// Backward compatibility for older model directories.
		modelFilename = "model_quantized.onnx"
	}

	// Required files
	requiredFiles := []string{
		modelFilename,
		"tokenizer.json",
		"label_mappings.json",
	}

	// Check for presence of all required files
	var missingFiles []string
	for _, filename := range requiredFiles {
		fullPath := filepath.Join(dir, filename)
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			missingFiles = append(missingFiles, filename)
		}
	}

	if len(missingFiles) > 0 {
		return nil, fmt.Errorf("missing required files in directory: %v", missingFiles)
	}

	// Return configuration with absolute paths
	absDir, err := filepath.Abs(dir)
	if err != nil {
		absDir = dir // Fall back to original if abs fails
	}

	config := &ModelConfig{
		ModelPath:     filepath.Join(absDir, modelFilename),
		TokenizerPath: filepath.Join(absDir, "tokenizer.json"),
		LabelMapPath:  filepath.Join(absDir, "label_mappings.json"),
	}

	log.Printf("[ModelManager] Validated directory: %s using %s", absDir, modelFilename)
	return config, nil
}

// Close closes the current detector and the extra detectors, cleaning up
// resources. It attempts to close all of them and returns the joined error.
func (mm *ModelManager) Close() error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	var errs []error
	if mm.currentDetector != nil {
		log.Printf("[ModelManager] Closing current detector")
		if err := mm.currentDetector.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close detector: %w", err))
		}
		mm.currentDetector = nil
	}

	for _, d := range mm.extraDetectors {
		if err := d.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close extra detector %q: %w", d.GetName(), err))
		}
	}
	mm.extraDetectors = nil

	mm.isHealthy = false
	return errors.Join(errs...)
}
