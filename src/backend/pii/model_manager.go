package pii

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

// ModelManager manages PII model lifecycle with thread-safe hot reload capability
type ModelManager struct {
	mu              sync.RWMutex
	currentDetector pii.Detector
	modelDirectory  string
	isHealthy       bool
	lastError       error
}

// ModelConfig holds paths to required model files
type ModelConfig struct {
	ModelPath     string
	TokenizerPath string
	LabelMapPath  string
}

// NewModelManager creates a new model manager and initializes with the given directory
func NewModelManager(directory string) (*ModelManager, error) {
	mm := &ModelManager{
		modelDirectory: directory,
		isHealthy:      false,
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

// GetDetector returns the current detector in a thread-safe manner
func (mm *ModelManager) GetDetector() (pii.Detector, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	if !mm.isHealthy {
		return nil, fmt.Errorf("model is unhealthy: %w", mm.lastError)
	}

	if mm.currentDetector == nil {
		return nil, fmt.Errorf("no detector available")
	}

	return mm.currentDetector, nil
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

	// Step 4: Swap detectors atomically (critical section)
	mm.mu.Lock()
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

// GetInfo returns information about the current model state
func (mm *ModelManager) GetInfo() map[string]interface{} {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	info := map[string]interface{}{
		"directory": mm.modelDirectory,
		"healthy":   mm.isHealthy,
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

	// Required files
	requiredFiles := []string{
		"model_quantized.onnx",
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
		ModelPath:     filepath.Join(absDir, "model_quantized.onnx"),
		TokenizerPath: filepath.Join(absDir, "tokenizer.json"),
		LabelMapPath:  filepath.Join(absDir, "label_mappings.json"),
	}

	log.Printf("[ModelManager] Validated directory: %s", absDir)
	return config, nil
}

// Close closes the current detector and cleans up resources
func (mm *ModelManager) Close() error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.currentDetector != nil {
		log.Printf("[ModelManager] Closing current detector")
		if err := mm.currentDetector.Close(); err != nil {
			return fmt.Errorf("failed to close detector: %w", err)
		}
		mm.currentDetector = nil
	}

	mm.isHealthy = false
	return nil
}
