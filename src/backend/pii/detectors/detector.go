package pii

import (
	"context"
	"fmt"
)

const (
	DetectorNameONNXModel = "onnx_model_detector"
)

type Detector interface {
	GetName() string
	Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error)
	Close() error
}

type NewDetectorFunc func(config map[string]interface{}) (Detector, error)

var detectorFactories = make(map[string]NewDetectorFunc)

func RegisterDetectorFactory(name string, factory NewDetectorFunc) {
	detectorFactories[name] = factory
}

func NewDetector(name string, config map[string]interface{}) (Detector, error) {
	factory, ok := detectorFactories[name]
	if !ok {
		return nil, fmt.Errorf("detector factory not found for name: %s", name)
	}
	return factory(config)
}

func init() {
	// Register built-in detector factories
	RegisterDetectorFactory(DetectorNameONNXModel, func(config map[string]interface{}) (Detector, error) {
		modelPath, ok := config["model_path"].(string)
		if !ok {
			return nil, fmt.Errorf("model_path is required for ONNX model detector")
		}
		tokenizerPath, ok := config["tokenizer_path"].(string)
		if !ok {
			return nil, fmt.Errorf("tokenizer_path is required for ONNX model detector")
		}
		return NewONNXModelDetectorSimple(modelPath, tokenizerPath)
	})
}

func CloseDetector(detector Detector) error {
	return detector.Close()
}
