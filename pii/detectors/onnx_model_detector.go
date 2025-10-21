package pii

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"

	"github.com/daulet/tokenizers"
	onnxruntime "github.com/yalue/onnxruntime_go"
)

// ONNXModelDetectorSimple implements DetectorClass using an internal ONNX model
type ONNXModelDetectorSimple struct {
	tokenizer    *tokenizers.Tokenizer
	session      *onnxruntime.AdvancedSession
	inputTensor  *onnxruntime.Tensor[int64]
	maskTensor   *onnxruntime.Tensor[int64]
	outputTensor *onnxruntime.Tensor[float32]
	id2label     map[string]string
	label2id     map[string]int
	modelPath    string
}

// NewONNXModelDetectorSimple creates a new ONNX model detector
func NewONNXModelDetectorSimple(modelPath string, tokenizerPath string) (*ONNXModelDetectorSimple, error) {
	// Set the ONNX Runtime shared library path for macOS
	onnxruntime.SetSharedLibraryPath("dist/yaak-proxy/libonnxruntime.1.23.1.dylib")

	// Initialize ONNX Runtime environment only if not already initialized
	if !onnxruntime.IsInitialized() {
		err := onnxruntime.InitializeEnvironment()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize ONNX Runtime environment: %w", err)
		}
	}

	// Load tokenizer
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		onnxruntime.DestroyEnvironment()
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Load model configuration
	configPath := "pii_onnx_model/config.json"
	configData, err := ioutil.ReadFile(configPath)
	if err != nil {
		tk.Close()
		onnxruntime.DestroyEnvironment()
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config struct {
		ID2Label map[string]string `json:"id2label"`
		Label2ID map[string]int    `json:"label2id"`
	}
	if err := json.Unmarshal(configData, &config); err != nil {
		tk.Close()
		onnxruntime.DestroyEnvironment()
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	detector := &ONNXModelDetectorSimple{
		tokenizer: tk,
		id2label:  config.ID2Label,
		label2id:  config.Label2ID,
		modelPath: modelPath,
	}

	// Initialize tensors and session will be done on first use
	return detector, nil
}

// GetName returns the name of this detector
func (d *ONNXModelDetectorSimple) GetName() string {
	return "onnx_model_detector_simple"
}

// Detect processes the input and returns detected entities
func (d *ONNXModelDetectorSimple) Detect(input DetectorInput) (DetectorOutput, error) {
	// Initialize session and tensors on first use
	if d.session == nil {
		if err := d.initializeSession(); err != nil {
			return DetectorOutput{}, fmt.Errorf("failed to initialize session: %w", err)
		}
	}

	// Tokenize input
	tokenIDs, _ := d.tokenizer.Encode(input.Text, true) // true = add special tokens

	// Convert to int64 for ONNX
	inputIDs := make([]int64, len(tokenIDs))
	attentionMask := make([]int64, len(tokenIDs))
	for i := range tokenIDs {
		inputIDs[i] = int64(tokenIDs[i])
		attentionMask[i] = 1 // All tokens are attended to
	}

	// Update input tensors with new data
	if err := d.updateInputTensors(inputIDs, attentionMask); err != nil {
		return DetectorOutput{}, fmt.Errorf("failed to update input tensors: %w", err)
	}

	// Run inference
	if err := d.session.Run(); err != nil {
		return DetectorOutput{}, fmt.Errorf("failed to run inference: %w", err)
	}

	// Process results inline to avoid the compilation issue
	entities := d.processOutputInline(input.Text, tokenIDs)

	return DetectorOutput{
		Text:     input.Text,
		Entities: entities,
	}, nil
}

// processOutputInline converts model output to entities (inline to avoid compilation issues)
func (d *ONNXModelDetectorSimple) processOutputInline(originalText string, tokenIDs []uint32) []Entity {
	outputData := d.outputTensor.GetData()
	entities := []Entity{}

	// Process each token
	for i := range tokenIDs {
		// Get logits for this token (33 classes)
		tokenLogits := outputData[i*33 : (i+1)*33]

		// Find the class with highest probability
		maxProb := float64(-math.MaxFloat64)
		bestClass := 0
		for j, logit := range tokenLogits {
			prob := float64(logit)
			if prob > maxProb {
				maxProb = prob
				bestClass = j
			}
		}

		// Convert class ID to label
		classID := fmt.Sprintf("%d", bestClass)
		label, exists := d.id2label[classID]
		if !exists {
			continue
		}

		// Skip "O" (outside) labels
		if label == "O" {
			continue
		}

		// Convert logits to probability (softmax)
		prob := math.Exp(maxProb)
		var sum float64
		for _, logit := range tokenLogits {
			sum += math.Exp(float64(logit))
		}
		confidence := prob / sum

		// Only include entities with reasonable confidence
		if confidence < 0.5 {
			continue
		}

		// For now, create a simple entity with token ID as text
		// TODO: Implement proper token-to-text mapping
		entities = append(entities, Entity{
			Text:       fmt.Sprintf("token_%d", tokenIDs[i]),
			Label:      label,
			StartPos:   i, // Simplified position
			EndPos:     i + 1,
			Confidence: confidence,
		})
	}

	return entities
}

// initializeSession initializes the ONNX session and tensors
func (d *ONNXModelDetectorSimple) initializeSession() error {
	// Create input tensors with maximum sequence length
	maxSeqLen := int64(512) // Based on config max_position_embeddings
	batchSize := int64(1)

	inputShape := onnxruntime.NewShape(batchSize, maxSeqLen)
	inputTensor, err := onnxruntime.NewTensor(inputShape, make([]int64, maxSeqLen))
	if err != nil {
		return fmt.Errorf("failed to create input tensor: %w", err)
	}

	maskTensor, err := onnxruntime.NewTensor(inputShape, make([]int64, maxSeqLen))
	if err != nil {
		inputTensor.Destroy()
		return fmt.Errorf("failed to create mask tensor: %w", err)
	}

	// Create output tensor
	outputShape := onnxruntime.NewShape(batchSize, maxSeqLen, 33) // 33 labels
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		maskTensor.Destroy()
		return fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Create session
	// d.modelPath already contains the full path to the model file
	session, err := onnxruntime.NewAdvancedSession(d.modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		[]onnxruntime.Value{inputTensor, maskTensor},
		[]onnxruntime.Value{outputTensor},
		nil)
	if err != nil {
		inputTensor.Destroy()
		maskTensor.Destroy()
		outputTensor.Destroy()
		return fmt.Errorf("failed to create session: %w", err)
	}

	d.session = session
	d.inputTensor = inputTensor
	d.maskTensor = maskTensor
	d.outputTensor = outputTensor

	return nil
}

// updateInputTensors updates the input tensors with new data
func (d *ONNXModelDetectorSimple) updateInputTensors(inputIDs, attentionMask []int64) error {
	// Get current tensor data and update it
	inputData := d.inputTensor.GetData()
	maskData := d.maskTensor.GetData()

	// Clear previous data
	for i := range inputData {
		inputData[i] = 0
		maskData[i] = 0
	}

	// Copy new data
	copy(inputData, inputIDs)
	copy(maskData, attentionMask)

	return nil
}

// Close implements the Detector interface
func (d *ONNXModelDetectorSimple) Close() error {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.inputTensor != nil {
		d.inputTensor.Destroy()
	}
	if d.maskTensor != nil {
		d.maskTensor.Destroy()
	}
	if d.outputTensor != nil {
		d.outputTensor.Destroy()
	}
	if d.tokenizer != nil {
		d.tokenizer.Close()
	}
	onnxruntime.DestroyEnvironment()
	return nil
}
