package pii

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/daulet/tokenizers"
	onnxruntime "github.com/yalue/onnxruntime_go"
)

// Chunking constants for processing long texts
const (
	maxSeqLen    = 512 // Maximum tokens per chunk (DistilBERT limit)
	chunkOverlap = 64  // Tokens of overlap between chunks for context continuity

	defaultEntityConfidenceThreshold = 0.25 // Default minimum confidence threshold for entity detection
)

// tokenChunk represents a chunk of tokens for processing
type tokenChunk struct {
	tokenIDs        []uint32
	offsets         []tokenizers.Offset
	startTokenIndex int  // Index of first token in original sequence
	isFirst         bool // Is this the first chunk?
	isLast          bool // Is this the last chunk?
}

// ONNXModelDetectorSimple implements DetectorClass using an internal ONNX model
type ONNXModelDetectorSimple struct {
	tokenizer                 *tokenizers.Tokenizer
	session                   *onnxruntime.AdvancedSession
	inputTensor               *onnxruntime.Tensor[int64]
	maskTensor                *onnxruntime.Tensor[int64]
	outputTensor              *onnxruntime.Tensor[float32]
	id2label                  map[string]string
	label2id                  map[string]int
	corefID2Label             map[string]string
	numPIILabels              int
	modelPath                 string
	entityConfidenceThreshold float64
}

// NewONNXModelDetectorSimple creates a new ONNX model detector
func NewONNXModelDetectorSimple(modelPath string, tokenizerPath string) (*ONNXModelDetectorSimple, error) {
	// Set the ONNX Runtime shared library path for macOS
	// Try multiple possible locations for the library
	onnxPaths := []string{
		"./libonnxruntime.1.23.1.dylib",       // Production: in resources directory
		"./build/libonnxruntime.1.23.1.dylib", // Development: in build directory
		"../libonnxruntime.1.23.1.dylib",      // Alternative location
	}

	var onnxLibPath string
	for _, path := range onnxPaths {
		if _, err := os.Stat(path); err == nil {
			onnxLibPath = path
			break
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
		// Don't destroy ONNX environment on error - it's shared globally
		// and may be in use by other detectors
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
		// Don't destroy ONNX environment on error - it's shared globally
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
		// Don't destroy ONNX environment on error - it's shared globally
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

	detector := &ONNXModelDetectorSimple{
		tokenizer:                 tk,
		id2label:                  config.PII.ID2Label,
		label2id:                  config.PII.Label2ID,
		corefID2Label:             config.Coref.ID2Label,
		numPIILabels:              numPIILabels,
		modelPath:                 modelPath,
		entityConfidenceThreshold: defaultEntityConfidenceThreshold,
	}

	// Initialize tensors and session will be done on first use
	return detector, nil
}

// GetName returns the name of this detector
func (d *ONNXModelDetectorSimple) GetName() string {
	return "onnx_model_detector_simple"
}

// SetEntityConfidenceThreshold updates the minimum confidence threshold for entity detection
func (d *ONNXModelDetectorSimple) SetEntityConfidenceThreshold(threshold float64) {
	d.entityConfidenceThreshold = threshold
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

	// Split tokens into chunks for processing long texts
	chunks := chunkTokens(tokenIDs, encoding.Offsets)

	// Process each chunk and collect entities
	var chunkEntities [][]Entity
	for _, chunk := range chunks {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return DetectorOutput{}, ctx.Err()
		default:
		}

		// Convert to int64 for ONNX
		inputIDs := make([]int64, len(chunk.tokenIDs))
		attentionMask := make([]int64, len(chunk.tokenIDs))
		for i := range chunk.tokenIDs {
			inputIDs[i] = int64(chunk.tokenIDs[i])
			attentionMask[i] = 1 // All tokens are attended to
		}

		// Update input tensors with new data
		d.updateInputTensors(inputIDs, attentionMask)

		// Run inference
		if err := d.session.Run(); err != nil {
			return DetectorOutput{}, fmt.Errorf("failed to run inference on chunk: %w", err)
		}

		// Process results for this chunk
		// Offsets in chunk already contain absolute character positions from original text
		entities := d.processOutputInline(input.Text, chunk.tokenIDs, chunk.offsets)
		chunkEntities = append(chunkEntities, entities)
	}

	// Merge entities from all chunks, handling overlaps
	entities := mergeChunkEntities(chunkEntities, input.Text)

	return DetectorOutput{
		Text:     input.Text,
		Entities: entities,
	}, nil
}

// classifyToken returns the best label and its softmax confidence for a single token's logits.
func (d *ONNXModelDetectorSimple) classifyToken(tokenLogits []float32) (string, float64) {
	maxProb := float64(-math.MaxFloat64)
	bestClass := 0
	for j, logit := range tokenLogits {
		prob := float64(logit)
		if prob > maxProb {
			maxProb = prob
			bestClass = j
		}
	}

	classID := fmt.Sprintf("%d", bestClass)
	label, exists := d.id2label[classID]
	if !exists {
		label = "O"
	}

	prob := math.Exp(maxProb)
	var sum float64
	for _, logit := range tokenLogits {
		sum += math.Exp(float64(logit))
	}
	confidence := prob / sum

	return label, confidence
}

// processOutputInline converts model output to entities (inline to avoid compilation issues)
func (d *ONNXModelDetectorSimple) processOutputInline(originalText string, tokenIDs []uint32, offsets []tokenizers.Offset) []Entity {
	outputData := d.outputTensor.GetData()
	entities := []Entity{}

	// Ensure we don't process more tokens than we have
	numTokens := len(tokenIDs)
	if len(offsets) < numTokens {
		numTokens = len(offsets)
	}

	fmt.Printf("[ONNX Model Response] Processing %d tokens, output tensor size: %d, numPIILabels: %d\n", numTokens, len(outputData), d.numPIILabels)

	// Group consecutive tokens with same label (B-PREFIX, I-PREFIX pattern)
	var currentEntity *Entity
	var currentTokens []int

	// Process each token
	for i := 0; i < numTokens; i++ {
		// Get logits for this token - ensure we don't go out of bounds
		startIdx := i * d.numPIILabels
		endIdx := (i + 1) * d.numPIILabels
		if endIdx > len(outputData) {
			fmt.Printf("[ONNX Model Response] Token %d: out of bounds (startIdx=%d, endIdx=%d, outputLen=%d)\n", i, startIdx, endIdx, len(outputData))
			break // Reached end of output data
		}
		tokenLogits := outputData[startIdx:endIdx]

		// Classify this token
		label, confidence := d.classifyToken(tokenLogits)

		// Log token details
		tokenText := ""
		if i < len(offsets) && offsets[i][0] < uint(len(originalText)) && offsets[i][1] <= uint(len(originalText)) {
			tokenText = originalText[offsets[i][0]:offsets[i][1]]
		}
		fmt.Printf("[ONNX Model Response] Token %d: id=%d text=%q offset=(%d,%d) label=%q confidence=%.4f\n",
			i, tokenIDs[i], tokenText, offsets[i][0], offsets[i][1], label, confidence)

		// Log top-3 predictions for non-O tokens to help debug misclassifications
		if label != "O" || confidence < 0.8 {
			type logitEntry struct {
				classID int
				label   string
				logit   float32
			}
			var top []logitEntry
			for j, logit := range tokenLogits {
				cID := fmt.Sprintf("%d", j)
				cLabel, ok := d.id2label[cID]
				if !ok {
					cLabel = fmt.Sprintf("UNKNOWN_%d", j)
				}
				top = append(top, logitEntry{j, cLabel, logit})
			}
			sort.Slice(top, func(a, b int) bool { return top[a].logit > top[b].logit })
			n := 3
			if len(top) < n {
				n = len(top)
			}
			fmt.Printf("[ONNX Model Response]   Top-%d predictions: ", n)
			for k := 0; k < n; k++ {
				fmt.Printf("%s(id=%d logit=%.4f) ", top[k].label, top[k].classID, top[k].logit)
			}
			fmt.Println()
		}

		// Only process tokens with reasonable confidence
		if confidence < d.entityConfidenceThreshold {
			label = "O"
		}

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

	fmt.Printf("[ONNX Model Response] Extracted %d entities from chunk:\n", len(entities))
	for i, e := range entities {
		fmt.Printf("[ONNX Model Response]   Entity %d: label=%q text=%q pos=(%d,%d) confidence=%.4f\n",
			i, e.Label, e.Text, e.StartPos, e.EndPos, e.Confidence)
	}

	return entities
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
	// Safe conversion with bounds checking
	const maxInt = int(^uint(0) >> 1)
	if startOffset[0] <= uint(maxInt) {
		// #nosec G115 - Safe conversion with bounds checking
		entity.StartPos = int(startOffset[0])
	} else {
		entity.StartPos = maxInt // Max int value
	}
	if endOffset[1] <= uint(maxInt) {
		// #nosec G115 - Safe conversion with bounds checking
		entity.EndPos = int(endOffset[1])
	} else {
		entity.EndPos = maxInt // Max int value
	}
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

	// Create output tensor
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

	// Create session
	// d.modelPath already contains the full path to the model file
	// Model outputs both pii_logits and coref_logits, but we only use pii_logits for now
	session, err := onnxruntime.NewAdvancedSession(d.modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"pii_logits"},
		[]onnxruntime.Value{inputTensor, maskTensor},
		[]onnxruntime.Value{outputTensor},
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
		return fmt.Errorf("failed to create session: %w", err)
	}

	d.session = session
	d.inputTensor = inputTensor
	d.maskTensor = maskTensor
	d.outputTensor = outputTensor

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
		d.session = nil
	}
	if d.inputTensor != nil {
		if err := d.inputTensor.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy input tensor: %w", err))
		}
		d.inputTensor = nil
	}
	if d.maskTensor != nil {
		if err := d.maskTensor.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy mask tensor: %w", err))
		}
		d.maskTensor = nil
	}
	if d.outputTensor != nil {
		if err := d.outputTensor.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("failed to destroy output tensor: %w", err))
		}
		d.outputTensor = nil
	}
	if d.tokenizer != nil {
		if err := d.tokenizer.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close tokenizer: %w", err))
		}
		d.tokenizer = nil
	}
	// NOTE: We intentionally do NOT call onnxruntime.DestroyEnvironment() here.
	// The ONNX runtime environment is global and shared across all detectors.
	// Destroying it would invalidate any other detectors that are still in use
	// (e.g., the new detector loaded during a hot reload).
	// The environment should only be destroyed when the entire application shuts down.

	if len(errs) > 0 {
		return fmt.Errorf("cleanup errors: %v", errs)
	}
	return nil
}

// chunkTokens splits tokens into overlapping chunks for processing long texts
func chunkTokens(tokenIDs []uint32, offsets []tokenizers.Offset) []tokenChunk {
	numTokens := len(tokenIDs)

	// If text fits in one chunk, return as-is
	if numTokens <= maxSeqLen {
		return []tokenChunk{{
			tokenIDs:        tokenIDs,
			offsets:         offsets,
			startTokenIndex: 0,
			isFirst:         true,
			isLast:          true,
		}}
	}

	var chunks []tokenChunk
	stride := maxSeqLen - chunkOverlap // 448 tokens per stride

	if stride <= 0 {
		panic("chunkOverlap must be less than maxSeqLen")
	}

	for start := 0; start < numTokens; start += stride {
		end := start + maxSeqLen
		if end > numTokens {
			end = numTokens
		}

		chunk := tokenChunk{
			tokenIDs:        tokenIDs[start:end],
			offsets:         offsets[start:end],
			startTokenIndex: start,
			isFirst:         start == 0,
			isLast:          end >= numTokens,
		}
		chunks = append(chunks, chunk)

		// Stop if we've reached the end
		if end >= numTokens {
			break
		}
	}

	return chunks
}

// mergeChunkEntities combines entities from multiple chunks, handling overlaps.
// The originalText parameter is used to correctly reconstruct text when merging
// overlapping entities, avoiding potential string corruption from naive concatenation.
func mergeChunkEntities(chunkEntities [][]Entity, originalText string) []Entity {
	if len(chunkEntities) == 0 {
		return []Entity{}
	}
	if len(chunkEntities) == 1 {
		return chunkEntities[0]
	}

	// Collect all entities
	var allEntities []Entity
	for _, entities := range chunkEntities {
		allEntities = append(allEntities, entities...)
	}

	if len(allEntities) == 0 {
		return []Entity{}
	}

	// Sort by start position
	sort.Slice(allEntities, func(i, j int) bool {
		if allEntities[i].StartPos != allEntities[j].StartPos {
			return allEntities[i].StartPos < allEntities[j].StartPos
		}
		// If same start, prefer longer entity
		return allEntities[i].EndPos > allEntities[j].EndPos
	})

	// Deduplicate overlapping entities (prefer higher confidence)
	var merged []Entity
	for _, entity := range allEntities {
		// Check if this entity overlaps with the last merged entity
		if len(merged) > 0 {
			last := &merged[len(merged)-1]

			// Check for overlap: entities overlap if one starts before the other ends
			if entity.StartPos < last.EndPos {
				// Overlapping entities - keep the one with higher confidence
				// or merge if they represent the same text span
				if entity.StartPos == last.StartPos && entity.EndPos == last.EndPos {
					// Exact same span - keep higher confidence
					if entity.Confidence > last.Confidence {
						*last = entity
					}
					continue
				}

				// Partial overlap - if same label, extend to cover both
				if entity.Label == last.Label {
					if entity.EndPos > last.EndPos {
						last.EndPos = entity.EndPos
						// Re-slice from original text to ensure correct merged text
						if last.StartPos < len(originalText) && last.EndPos <= len(originalText) {
							last.Text = originalText[last.StartPos:last.EndPos]
						}
						last.Confidence = (last.Confidence + entity.Confidence) / 2
					}
					continue
				}

				// Different labels with overlap - keep higher confidence one
				if entity.Confidence > last.Confidence {
					*last = entity
				}
				continue
			}
		}

		// No overlap, add as new entity
		merged = append(merged, entity)
	}

	return merged
}
