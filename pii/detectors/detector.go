package pii

import "fmt"

const (
	DetectorNameModel     = "model_detector"
	DetectorNameRegex     = "regex_detector"
	DetectorNameONNXModel = "onnx_model_detector"
)

type Detector interface {
	GetName() string
	Detect(input DetectorInput) (DetectorOutput, error)
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
	RegisterDetectorFactory(DetectorNameModel, func(config map[string]interface{}) (Detector, error) {
		baseURL, ok := config["base_url"].(string)
		if !ok {
			return nil, fmt.Errorf("base_url is required for model detector")
		}
		return NewModelDetector(baseURL), nil
	})

	RegisterDetectorFactory(DetectorNameRegex, func(config map[string]interface{}) (Detector, error) {
		return NewRegexDetector(PIIPatterns), nil
	})

	RegisterDetectorFactory(DetectorNameONNXModel, func(config map[string]interface{}) (Detector, error) {
		modelPath, ok := config["model_path"].(string)
		if !ok {
			return nil, fmt.Errorf("model_path is required for ONNX model detector")
		}
		return NewONNXModelDetectorSimple(modelPath)
	})
}

func CloseDetector(detector Detector) error {
	return detector.Close()
}
