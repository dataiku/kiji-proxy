package pii

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/daulet/tokenizers"
	onnxruntime "github.com/yalue/onnxruntime_go"
)

// ONNXModelDetectorSimple implements DetectorClass using an internal ONNX model
type ONNXModelDetectorSimple struct {
	tokenizer         *tokenizers.Tokenizer
	session           *onnxruntime.AdvancedSession
	inputTensor       *onnxruntime.Tensor[int64]
	maskTensor        *onnxruntime.Tensor[int64]
	outputTensor      *onnxruntime.Tensor[float32]
	corefOutputTensor *onnxruntime.Tensor[float32]
	id2label          map[string]string
	label2id          map[string]int
	corefID2Label     map[string]string
	numPIILabels      int
	numCorefLabels    int
	modelPath         string
}

// safeUintToInt safely converts a uint to int with bounds checking
// Returns maxInt if the value would overflow
func safeUintToInt(val uint) int {
	const maxInt = int(^uint(0) >> 1)
	if val <= uint(maxInt) {
		// #nosec G115 - Safe conversion with bounds checking
		return int(val)
	}
	return maxInt
}

// NewONNXModelDetectorSimple creates a new ONNX model detector
func NewONNXModelDetectorSimple(modelPath string, tokenizerPath string) (*ONNXModelDetectorSimple, error) {
	// Set the ONNX Runtime shared library path for macOS
	// 1. Check if environment variable is set
	onnxLibPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")

	// 2. If not set, try multiple possible locations and versions
	if onnxLibPath == "" {
		onnxPaths := []string{
			"./libonnxruntime.1.23.1.dylib",       // Production: in resources directory
			"./build/libonnxruntime.1.23.1.dylib", // Development: in build directory
			"./libonnxruntime.1.23.2.dylib",       // Newer version
			"./build/libonnxruntime.1.23.2.dylib", // Newer version in build
			"../libonnxruntime.1.23.1.dylib",      // Alternative location
		}

		for _, path := range onnxPaths {
			if _, err := os.Stat(path); err == nil {
				onnxLibPath = path
				break
			}
		}
	}

	if onnxLibPath != "" {
		onnxruntime.SetSharedLibraryPath(onnxLibPath)
	} else {
		// Fall back to default path, might work if library is in system path
		onnxruntime.SetSharedLibraryPath("./build/libonnxruntime.1.23.1.dylib")
	}

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
		if err := onnxruntime.DestroyEnvironment(); err != nil {
			// Log but don't fail on cleanup error
			fmt.Printf("Warning: failed to destroy environment during cleanup: %v\n", err)
		}
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	// Load model configuration
	// Try multiple possible locations for the config file
	configPaths := []string{
		"model/quantized/label_mappings.json", // Default location
		"quantized/label_mappings.json",       // Alternative: in resources/quantized
		"./label_mappings.json",               // Alternative: current directory
	}

	var configData []byte
	for _, path := range configPaths {
		data, err := os.ReadFile(path)
		if err == nil {
			configData = data
			break
		}
	}

	if configData == nil {
		if err := tk.Close(); err != nil {
			fmt.Printf("Warning: failed to close tokenizer during cleanup: %v\n", err)
		}
		if err := onnxruntime.DestroyEnvironment(); err != nil {
			fmt.Printf("Warning: failed to destroy environment during cleanup: %v\n", err)
		}
		return nil, fmt.Errorf("failed to load model configuration from any of the attempted paths: %v", configPaths)
	}

	var config struct {
		PII struct {
			ID2Label map[string]string `json:"id2label"`
			Label2ID map[string]int    `json:"label2id"`
		} `json:"pii"`
		Coref struct {
			ID2Label map[string]string `json:"id2label"`
		} `json:"coref"`
	}
	if err := json.Unmarshal(configData, &config); err != nil {
		if err := tk.Close(); err != nil {
			fmt.Printf("Warning: failed to close tokenizer during cleanup: %v\n", err)
		}
		if err := onnxruntime.DestroyEnvironment(); err != nil {
			fmt.Printf("Warning: failed to destroy environment during cleanup: %v\n", err)
		}
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Calculate number of PII labels from the id2label mapping
	// Find the maximum label ID and add 1 (since IDs are 0-indexed)
	numPIILabels := 0
	for idStr := range config.PII.ID2Label {
		// Skip special labels like "-100" for IGNORE
		if idStr == "-100" {
			continue
		}
		var id int
		if _, err := fmt.Sscanf(idStr, "%d", &id); err == nil {
			if id >= numPIILabels {
				numPIILabels = id + 1
			}
		}
	}
	if numPIILabels == 0 {
		// Fallback: use label2id count if id2label parsing fails
		numPIILabels = len(config.PII.Label2ID)
	}
	fmt.Printf("Loaded %d PII labels (expected 49)\n", numPIILabels)
	// Calculate number of Coref labels
	numCorefLabels := 0
	for idStr := range config.Coref.ID2Label {
		var id int
		if _, err := fmt.Sscanf(idStr, "%d", &id); err == nil {
			if id >= numCorefLabels {
				numCorefLabels = id + 1
			}
		}
	}
	if numCorefLabels == 0 {
		numCorefLabels = len(config.Coref.ID2Label)
	}

	detector := &ONNXModelDetectorSimple{
		tokenizer:      tk,
		id2label:       config.PII.ID2Label,
		label2id:       config.PII.Label2ID,
		corefID2Label:  config.Coref.ID2Label,
		numPIILabels:   numPIILabels,
		numCorefLabels: numCorefLabels,
		modelPath:      modelPath,
	}

	// Initialize tensors and session will be done on first use
	return detector, nil
}

// GetName returns the name of this detector
func (d *ONNXModelDetectorSimple) GetName() string {
	return "onnx_model_detector_simple"
}

// Detect processes the input and returns detected entities
func (d *ONNXModelDetectorSimple) Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error) {
	// Initialize session and tensors on first use
	if d.session == nil {
		if err := d.initializeSession(); err != nil {
			return DetectorOutput{}, fmt.Errorf("failed to initialize session: %w", err)
		}
	}

	// Tokenize input with offsets to get character positions
	encoding := d.tokenizer.EncodeWithOptions(input.Text, true, tokenizers.WithReturnOffsets())
	tokenIDs := encoding.IDs

	// Convert to int64 for ONNX
	inputIDs := make([]int64, len(tokenIDs))
	attentionMask := make([]int64, len(tokenIDs))
	for i := range tokenIDs {
		inputIDs[i] = int64(tokenIDs[i])
		attentionMask[i] = 1 // All tokens are attended to
	}

	// Update input tensors with new data
	d.updateInputTensors(inputIDs, attentionMask)

	// Run inference
	if err := d.session.Run(); err != nil {
		return DetectorOutput{}, fmt.Errorf("failed to run inference: %w", err)
	}

	// Process results inline to avoid the compilation issue
	entities, corefClusters := d.processOutputInline(input.Text, tokenIDs, encoding.Offsets)
	for _, entity := range entities {
		fmt.Printf("Detected entity: %s, Label: %s, Confidence: %.4f, Range: [%d:%d]\n",
			entity.Text, entity.Label, entity.Confidence, entity.StartPos, entity.EndPos)
	}

	for clusterID, mentions := range corefClusters {
		fmt.Printf("Coref cluster %d: %v\n", clusterID, mentions)
	}

	return DetectorOutput{
		Text:          input.Text,
		Entities:      entities,
		CorefClusters: corefClusters,
	}, nil
}

// processOutputInline converts model output to entities and coref clusters
func (d *ONNXModelDetectorSimple) processOutputInline(originalText string, tokenIDs []uint32, offsets []tokenizers.Offset) ([]Entity, map[int][]EntityMention) {
	outputData := d.outputTensor.GetData()
	corefData := d.corefOutputTensor.GetData()
	entities := []Entity{}
	corefClusters := make(map[int][]EntityMention)

	// Ensure we don't process more tokens than we have
	numTokens := len(tokenIDs)
	if len(offsets) < numTokens {
		numTokens = len(offsets)
	}

	// Group consecutive tokens with same label (B-PREFIX, I-PREFIX pattern)
	var currentEntity *Entity
	var currentTokens []int

	// Process each token
	for i := 0; i < numTokens; i++ {
		// Get logits for this token - ensure we don't go out of bounds
		startIdx := i * d.numPIILabels
		endIdx := (i + 1) * d.numPIILabels
		if endIdx > len(outputData) {
			break // Reached end of output data
		}
		tokenLogits := outputData[startIdx:endIdx]

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
			label = "O"
		}

		// Convert logits to probability (softmax)
		prob := math.Exp(maxProb)
		var sum float64
		for _, logit := range tokenLogits {
			sum += math.Exp(float64(logit))
		}
		confidence := prob / sum

		// Only process tokens with reasonable confidence
		if confidence < 0.5 {
			label = "O"
		}

		// --- Coref Extraction ---
		corefStartIdx := i * d.numCorefLabels
		corefEndIdx := (i + 1) * d.numCorefLabels
		if corefEndIdx <= len(corefData) {
			tokenCorefLogits := corefData[corefStartIdx:corefEndIdx]
			maxCorefProb := float32(-math.MaxFloat32)
			bestCluster := 0
			for j, logit := range tokenCorefLogits {
				if logit > maxCorefProb {
					maxCorefProb = logit
					bestCluster = j
				}
			}

			if bestCluster > 0 { // Skip cluster 0
				startOffset := offsets[i]
				mention := EntityMention{
					Text:     originalText[startOffset[0]:startOffset[1]],
					StartPos: safeUintToInt(startOffset[0]),
					EndPos:   safeUintToInt(startOffset[1]),
					IsEntity: label != "O",
				}
				corefClusters[bestCluster] = append(corefClusters[bestCluster], mention)
			}
		}
		// ------------------------

		// Handle B-PREFIX (beginning) and I-PREFIX (inside) labels
		isBeginning := strings.HasPrefix(label, "B-")
		isInside := strings.HasPrefix(label, "I-")
		baseLabel := label
		if isBeginning || isInside {
			baseLabel = strings.TrimPrefix(strings.TrimPrefix(label, "B-"), "I-")
		}

		// Handle different entity states using switch for better readability
		switch {
		case label != "O" && (isBeginning || currentEntity == nil):
			// Finish previous entity if exists
			if currentEntity != nil {
				d.finalizeEntity(currentEntity, currentTokens, originalText, offsets)
				entities = append(entities, *currentEntity)
			}

			// Start new entity
			currentEntity = &Entity{
				Label:      baseLabel,
				Confidence: confidence,
			}
			currentTokens = []int{i}
		case label != "O" && isInside && currentEntity != nil && currentEntity.Label == baseLabel:
			// Continue current entity
			currentTokens = append(currentTokens, i)
			// Update confidence to average
			currentEntity.Confidence = (currentEntity.Confidence + confidence) / 2
		default:
			// Finish current entity if exists
			if currentEntity != nil {
				d.finalizeEntity(currentEntity, currentTokens, originalText, offsets)
				entities = append(entities, *currentEntity)
				currentEntity = nil
				currentTokens = nil
			}
		}
	}

	// Finish last entity if exists
	if currentEntity != nil {
		d.finalizeEntity(currentEntity, currentTokens, originalText, offsets)
		entities = append(entities, *currentEntity)
	}

	// Assign cluster IDs to entities
	for i := range entities {
		entities[i].ClusterID = d.findClusterForEntity(&entities[i], corefClusters)
	}

	return entities, corefClusters
}

// findClusterForEntity finds the best matching cluster ID for a detected PII entity
func (d *ONNXModelDetectorSimple) findClusterForEntity(entity *Entity, clusters map[int][]EntityMention) int {
	for clusterID, mentions := range clusters {
		for _, mention := range mentions {
			// If our entity perfectly overlaps or contains/is contained by a mention that is marked as isEntity
			if mention.StartPos == entity.StartPos && mention.EndPos == entity.EndPos {
				return clusterID
			}
		}
	}
	return 0
}

// finalizeEntity extracts the actual text from the original string using token offsets
func (d *ONNXModelDetectorSimple) finalizeEntity(entity *Entity, tokenIndices []int, originalText string, offsets []tokenizers.Offset) {
	if len(tokenIndices) == 0 {
		return
	}

	// Get the start and end character positions
	startOffset := offsets[tokenIndices[0]]
	endOffset := offsets[tokenIndices[len(tokenIndices)-1]]

	// Extract the actual text from the original string
	entity.Text = originalText[startOffset[0]:endOffset[1]]
	entity.StartPos = safeUintToInt(startOffset[0])
	entity.EndPos = safeUintToInt(endOffset[1])
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
		if err := inputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy input tensor during cleanup: %v\n", err)
		}
		return fmt.Errorf("failed to create mask tensor: %w", err)
	}

	// Create output tensors
	outputShape := onnxruntime.NewShape(batchSize, maxSeqLen, int64(d.numPIILabels))
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy input tensor during cleanup: %v\n", err)
		}
		if err := maskTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy mask tensor during cleanup: %v\n", err)
		}
		return fmt.Errorf("failed to create output tensor: %w", err)
	}

	corefOutputShape := onnxruntime.NewShape(batchSize, maxSeqLen, int64(d.numCorefLabels))
	corefOutputTensor, err := onnxruntime.NewEmptyTensor[float32](corefOutputShape)
	if err != nil {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy input tensor during cleanup: %v\n", err)
		}
		if err := maskTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy mask tensor during cleanup: %v\n", err)
		}
		if err := outputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy output tensor during cleanup: %v\n", err)
		}
		return fmt.Errorf("failed to create coref output tensor: %w", err)
	}

	// Create session
	// d.modelPath already contains the full path to the model file
	// Model outputs both pii_logits and coref_logits, but we now use both
	session, err := onnxruntime.NewAdvancedSession(d.modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"pii_logits", "coref_logits"},
		[]onnxruntime.Value{inputTensor, maskTensor},
		[]onnxruntime.Value{outputTensor, corefOutputTensor},
		nil)
	if err != nil {
		if err := inputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy input tensor during cleanup: %v\n", err)
		}
		if err := maskTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy mask tensor during cleanup: %v\n", err)
		}
		if err := outputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy output tensor during cleanup: %v\n", err)
		}
		if err := corefOutputTensor.Destroy(); err != nil {
			fmt.Printf("Warning: failed to destroy coref output tensor during cleanup: %v\n", err)
		}
		return fmt.Errorf("failed to create session: %w", err)
	}

	d.session = session
	d.inputTensor = inputTensor
	d.maskTensor = maskTensor
	d.outputTensor = outputTensor
	d.corefOutputTensor = corefOutputTensor

	return nil
}

// updateInputTensors updates the input tensors with new data
func (d *ONNXModelDetectorSimple) updateInputTensors(inputIDs, attentionMask []int64) {
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
}

// Close implements the Detector interface
func (d *ONNXModelDetectorSimple) Close() error {
	var errs []error

	if d.session != nil {
		if err := d.session.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy session: %w", err))
		}
	}
	if d.inputTensor != nil {
		if err := d.inputTensor.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy input tensor: %w", err))
		}
	}
	if d.maskTensor != nil {
		if err := d.maskTensor.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy mask tensor: %w", err))
		}
	}
	if d.outputTensor != nil {
		if err := d.outputTensor.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy output tensor: %w", err))
		}
	}
	if d.tokenizer != nil {
		if err := d.tokenizer.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close tokenizer: %w", err))
		}
	}
	if err := onnxruntime.DestroyEnvironment(); err != nil {
		errs = append(errs, fmt.Errorf("failed to destroy environment: %w", err))
	}

	if len(errs) > 0 {
		return fmt.Errorf("cleanup errors: %v", errs)
	}
	return nil
}
