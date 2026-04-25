package pii

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
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

// crfParams holds the CRF transition matrices exported during quantization.
// These are used by Viterbi decoding to enforce valid BIO label sequences.
type crfParams struct {
	Transitions      [][]float32 `json:"transitions"`       // [num_labels][num_labels]
	StartTransitions []float32   `json:"start_transitions"` // [num_labels]
	EndTransitions   []float32   `json:"end_transitions"`   // [num_labels]
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
	crf                       *crfParams // nil if crf_transitions.json not found
}

// NewONNXModelDetectorSimple creates a new ONNX model detector
func NewONNXModelDetectorSimple(modelPath string, tokenizerPath string) (*ONNXModelDetectorSimple, error) {
	// Set the ONNX Runtime shared library path.
	// Check ONNXRUNTIME_SHARED_LIBRARY_PATH env var first (set by Electron),
	// then try multiple possible locations for the library.
	onnxLibPath := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if onnxLibPath != "" {
		if _, err := os.Stat(onnxLibPath); err != nil {
			onnxLibPath = "" // env var path doesn't exist, fall through to search
		}
	}

	if onnxLibPath == "" {
		onnxPaths := []string{
			// macOS paths (.dylib)
			"./libonnxruntime.1.24.2.dylib",           // CWD (legacy)
			"./resources/libonnxruntime.1.24.2.dylib", // Production DMG: CWD is Contents/Resources
			"./build/libonnxruntime.1.24.2.dylib",     // Development: in build directory
			"../libonnxruntime.1.24.2.dylib",          // Alternative location
			// Linux paths (.so)
			"./lib/libonnxruntime.so.1.24.2",   // Linux release tarball layout
			"./build/libonnxruntime.so.1.24.2", // Development: in build directory
			"./libonnxruntime.so.1.24.2",       // CWD
		}

		for _, p := range onnxPaths {
			if _, err := os.Stat(p); err == nil {
				onnxLibPath = p
				break
			}
		}
	}

	if onnxLibPath != "" {
		onnxruntime.SetSharedLibraryPath(onnxLibPath)
	} else {
		// Fall back to default path, might work if library is in system path
		if runtime.GOOS == "linux" {
			onnxruntime.SetSharedLibraryPath("./lib/libonnxruntime.so.1.24.2")
		} else {
			onnxruntime.SetSharedLibraryPath("./build/libonnxruntime.1.24.2.dylib")
		}
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

	// Load CRF transition parameters for Viterbi decoding.
	// The CRF layer is part of the trained model but not exported to ONNX;
	// instead the transition matrices are saved as a sidecar JSON file.
	var crf *crfParams
	crfPaths := []string{
		"model/quantized/crf_transitions.json",
		"quantized/crf_transitions.json",
		"./crf_transitions.json",
	}
	for _, path := range crfPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var c crfParams
		if err := json.Unmarshal(data, &c); err != nil {
			fmt.Printf("Warning: failed to parse CRF transitions from %s: %v\n", path, err)
			continue
		}
		crf = &c
		fmt.Printf("Loaded CRF transition parameters from %s (%d labels)\n", path, len(c.StartTransitions))
		break
	}
	if crf == nil {
		fmt.Println("Warning: CRF transitions not found, falling back to argmax decoding")
	}

	detector := &ONNXModelDetectorSimple{
		tokenizer:                 tk,
		id2label:                  config.PII.ID2Label,
		label2id:                  config.PII.Label2ID,
		corefID2Label:             config.Coref.ID2Label,
		numPIILabels:              numPIILabels,
		modelPath:                 modelPath,
		entityConfidenceThreshold: defaultEntityConfidenceThreshold,
		crf:                       crf,
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

	// Tokenize input with offsets to get character positions.
	// Note: truncation is disabled in tokenizer.json (set to null) so that
	// long texts are not silently truncated. We handle chunking ourselves.
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

// viterbiDecode runs the Viterbi algorithm over a sequence of emission scores
// using the CRF transition matrices to find the globally optimal label sequence.
func viterbiDecode(emissions []float32, numLabels int, crf *crfParams) []int {
	seqLen := len(emissions) / numLabels
	if seqLen == 0 || numLabels == 0 {
		return nil
	}
	if len(emissions) < seqLen*numLabels {
		return nil
	}

	// viterbi[t*numLabels + j] = best score ending in label j at position t
	viterbi := make([]float64, seqLen*numLabels)
	backpointers := make([]int, seqLen*numLabels)

	// Initialization: viterbi[0][j] = start_transitions[j] + emissions[0][j]
	for j := 0; j < numLabels; j++ {
		viterbi[j] = float64(crf.StartTransitions[j]) + float64(emissions[j]) // #nosec G602 - j < numLabels <= len(emissions) checked above
	}

	// Forward pass
	for t := 1; t < seqLen; t++ {
		for j := 0; j < numLabels; j++ {
			bestScore := math.Inf(-1)
			bestPrev := 0
			emissionScore := float64(emissions[t*numLabels+j])
			for k := 0; k < numLabels; k++ {
				score := viterbi[(t-1)*numLabels+k] + float64(crf.Transitions[k][j]) + emissionScore
				if score > bestScore {
					bestScore = score
					bestPrev = k
				}
			}
			viterbi[t*numLabels+j] = bestScore
			backpointers[t*numLabels+j] = bestPrev
		}
	}

	// Add end transitions and find best last label
	bestScore := math.Inf(-1)
	bestLast := 0
	for j := 0; j < numLabels; j++ {
		score := viterbi[(seqLen-1)*numLabels+j] + float64(crf.EndTransitions[j])
		if score > bestScore {
			bestScore = score
			bestLast = j
		}
	}

	// Backtrace
	path := make([]int, seqLen)
	path[seqLen-1] = bestLast
	for t := seqLen - 1; t > 0; t-- {
		path[t-1] = backpointers[t*numLabels+path[t]]
	}

	return path
}

// softmaxConfidence computes the softmax probability for a specific class given its logits.
func softmaxConfidence(tokenLogits []float32, classIdx int) float64 {
	maxLogit := float64(tokenLogits[0])
	for _, l := range tokenLogits[1:] {
		if float64(l) > maxLogit {
			maxLogit = float64(l)
		}
	}
	var sum float64
	for _, l := range tokenLogits {
		sum += math.Exp(float64(l) - maxLogit)
	}
	return math.Exp(float64(tokenLogits[classIdx])-maxLogit) / sum
}

func splitBIOLabel(label string) (baseLabel string, isBeginning bool, isInside bool) {
	isBeginning = strings.HasPrefix(label, "B-")
	isInside = strings.HasPrefix(label, "I-")
	if isBeginning || isInside {
		return strings.TrimPrefix(strings.TrimPrefix(label, "B-"), "I-"), isBeginning, isInside
	}
	return label, false, false
}

func isEntityJoinerToken(tokenText string) bool {
	trimmed := strings.TrimSpace(tokenText)
	if trimmed == "" {
		return false
	}
	for _, r := range trimmed {
		if !strings.ContainsRune(".,@_-+:/#%&=", r) {
			return false
		}
	}
	return true
}

func tokenStartsAtPreviousEnd(tokenIndex int, currentTokens []int, offsets []tokenizers.Offset) bool {
	if len(currentTokens) == 0 || tokenIndex >= len(offsets) {
		return false
	}
	previousTokenIndex := currentTokens[len(currentTokens)-1]
	if previousTokenIndex >= len(offsets) {
		return false
	}
	return offsets[previousTokenIndex][1] == offsets[tokenIndex][0]
}

func tokenTextAt(originalText string, offsets []tokenizers.Offset, index int) string {
	if index >= len(offsets) || offsets[index][0] >= uint(len(originalText)) || offsets[index][1] > uint(len(originalText)) {
		return ""
	}
	return originalText[offsets[index][0]:offsets[index][1]]
}

func (d *ONNXModelDetectorSimple) decodedTokenLabel(index int, outputData []float32, bestLabels []int) (string, float64) {
	startIdx := index * d.numPIILabels
	endIdx := (index + 1) * d.numPIILabels
	if startIdx < 0 || endIdx > len(outputData) {
		return "O", 0
	}

	tokenLogits := outputData[startIdx:endIdx]
	if bestLabels != nil && index < len(bestLabels) {
		classID := bestLabels[index]
		idStr := fmt.Sprintf("%d", classID)
		label, ok := d.id2label[idStr]
		if !ok {
			label = "O"
		}
		return label, softmaxConfidence(tokenLogits, classID)
	}

	return d.classifyToken(tokenLogits)
}

type decodedEntityToken struct {
	index       int
	tokenID     uint32
	text        string
	label       string
	baseLabel   string
	confidence  float64
	isBeginning bool
	isInside    bool
}

func (d *ONNXModelDetectorSimple) decodeEntityToken(
	index int,
	tokenID uint32,
	originalText string,
	offsets []tokenizers.Offset,
	outputData []float32,
	bestLabels []int,
	currentEntity *Entity,
) decodedEntityToken {
	label, confidence := d.decodedTokenLabel(index, outputData, bestLabels)
	tokenText := tokenTextAt(originalText, offsets, index)

	if confidence < d.entityConfidenceThreshold {
		label = "O"
	}
	label, confidence = d.bridgeJoinerToken(index, tokenText, label, confidence, outputData, bestLabels, currentEntity)

	baseLabel, isBeginning, isInside := splitBIOLabel(label)
	return decodedEntityToken{
		index:       index,
		tokenID:     tokenID,
		text:        tokenText,
		label:       label,
		baseLabel:   baseLabel,
		confidence:  confidence,
		isBeginning: isBeginning,
		isInside:    isInside,
	}
}

func (d *ONNXModelDetectorSimple) bridgeJoinerToken(
	index int,
	tokenText string,
	label string,
	confidence float64,
	outputData []float32,
	bestLabels []int,
	currentEntity *Entity,
) (string, float64) {
	if label != "O" || currentEntity == nil || !isEntityJoinerToken(tokenText) {
		return label, confidence
	}

	nextLabel, nextConfidence := d.decodedTokenLabel(index+1, outputData, bestLabels)
	nextBaseLabel, nextIsBeginning, nextIsInside := splitBIOLabel(nextLabel)
	if nextConfidence >= d.entityConfidenceThreshold &&
		(nextIsBeginning || nextIsInside) &&
		nextBaseLabel == currentEntity.Label {
		return "I-" + currentEntity.Label, currentEntity.Confidence
	}

	return label, confidence
}

type entityGroupingState struct {
	currentEntity *Entity
	currentTokens []int
	entities      []Entity
}

func (s *entityGroupingState) consume(
	d *ONNXModelDetectorSimple,
	token decodedEntityToken,
	originalText string,
	offsets []tokenizers.Offset,
) {
	isSameCompactEntity := token.label != "O" &&
		s.currentEntity != nil &&
		s.currentEntity.Label == token.baseLabel &&
		tokenStartsAtPreviousEnd(token.index, s.currentTokens, offsets)

	switch {
	case token.label != "O" && s.currentEntity != nil && s.currentEntity.Label == token.baseLabel && (token.isInside || isSameCompactEntity):
		// Continue current entity. Same-label B tags immediately after a
		// joiner token are treated as continuation so emails like
		// "jonathan.reyes@example.com" do not fragment.
		s.currentTokens = append(s.currentTokens, token.index)
		s.currentEntity.Confidence = (s.currentEntity.Confidence + token.confidence) / 2
	case token.label != "O" && (token.isBeginning || s.currentEntity == nil):
		s.finishCurrent(d, originalText, offsets)
		s.currentEntity = &Entity{
			Label:      token.baseLabel,
			Confidence: token.confidence,
		}
		s.currentTokens = []int{token.index}
	default:
		s.finishCurrent(d, originalText, offsets)
	}
}

func (s *entityGroupingState) finishCurrent(d *ONNXModelDetectorSimple, originalText string, offsets []tokenizers.Offset) {
	if s.currentEntity == nil {
		return
	}

	d.finalizeEntity(s.currentEntity, s.currentTokens, originalText, offsets)
	s.entities = append(s.entities, *s.currentEntity)
	s.currentEntity = nil
	s.currentTokens = nil
}

func filterEmptyEntities(entities []Entity) []Entity {
	filtered := entities[:0]
	for _, e := range entities {
		if e.Text != "" {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

// processOutputInline converts model output to entities (inline to avoid compilation issues)
func (d *ONNXModelDetectorSimple) processOutputInline(originalText string, tokenIDs []uint32, offsets []tokenizers.Offset) []Entity {
	outputData := d.outputTensor.GetData()

	// Ensure we don't process more tokens than we have
	numTokens := len(tokenIDs)
	if len(offsets) < numTokens {
		numTokens = len(offsets)
	}

	fmt.Printf("[ONNX Model Response] Processing %d tokens, output tensor size: %d, numPIILabels: %d\n", numTokens, len(outputData), d.numPIILabels)

	// Decode label sequence: use Viterbi if CRF params available, otherwise argmax
	var bestLabels []int
	if d.crf != nil && numTokens*d.numPIILabels <= len(outputData) {
		bestLabels = viterbiDecode(outputData[:numTokens*d.numPIILabels], d.numPIILabels, d.crf)
	}

	// Group consecutive tokens with same label (B-PREFIX, I-PREFIX pattern)
	state := entityGroupingState{}

	// Process each token
	for i := 0; i < numTokens; i++ {
		// Get logits for this token - ensure we don't go out of bounds
		startIdx := i * d.numPIILabels
		endIdx := (i + 1) * d.numPIILabels
		if endIdx > len(outputData) {
			fmt.Printf("[ONNX Model Response] Token %d: out of bounds (startIdx=%d, endIdx=%d, outputLen=%d)\n", i, startIdx, endIdx, len(outputData))
			break // Reached end of output data
		}

		token := d.decodeEntityToken(i, tokenIDs[i], originalText, offsets, outputData, bestLabels, state.currentEntity)
		fmt.Printf("[ONNX Model Response] Token %d: id=%d text=%q offset=(%d,%d) label=%q confidence=%.4f\n",
			i, token.tokenID, token.text, offsets[i][0], offsets[i][1], token.label, token.confidence)

		state.consume(d, token, originalText, offsets)
	}

	// Finish last entity if exists
	state.finishCurrent(d, originalText, offsets)

	// Filter out entities with empty text (e.g. from special tokens like [CLS]/[SEP])
	entities := filterEmptyEntities(state.entities)

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

	// Trim leading/trailing whitespace from entity text and adjust offsets.
	// The SentencePiece Metaspace pre-tokenizer includes the preceding space
	// character in token offsets.
	trimmedStart := startOffset[0]
	trimmedEnd := endOffset[1]
	for trimmedStart < trimmedEnd && (originalText[trimmedStart] == ' ' || originalText[trimmedStart] == '\t' || originalText[trimmedStart] == '\n' || originalText[trimmedStart] == '\r') {
		trimmedStart++
	}
	for trimmedEnd > trimmedStart && (originalText[trimmedEnd-1] == ' ' || originalText[trimmedEnd-1] == '\t' || originalText[trimmedEnd-1] == '\n' || originalText[trimmedEnd-1] == '\r') {
		trimmedEnd--
	}
	// Strip trailing sentence punctuation only when followed by whitespace
	// or end-of-string, so "yahoo.com" keeps the dot but "1988," loses
	// the trailing comma.
	for trimmedEnd > trimmedStart && (originalText[trimmedEnd-1] == ',' || originalText[trimmedEnd-1] == '.' || originalText[trimmedEnd-1] == ';' || originalText[trimmedEnd-1] == ':' || originalText[trimmedEnd-1] == '!' || originalText[trimmedEnd-1] == '?') {
		if trimmedEnd < uint(len(originalText)) && originalText[trimmedEnd] != ' ' && originalText[trimmedEnd] != '\t' && originalText[trimmedEnd] != '\n' && originalText[trimmedEnd] != '\r' {
			break
		}
		trimmedEnd--
	}
	if trimmedStart < trimmedEnd {
		entity.Text = originalText[trimmedStart:trimmedEnd]
	} else {
		entity.Text = ""
	}

	// Safe conversion with bounds checking
	const maxInt = int(^uint(0) >> 1)
	if trimmedStart <= uint(maxInt) {
		// #nosec G115 - Safe conversion with bounds checking
		entity.StartPos = int(trimmedStart)
	} else {
		entity.StartPos = maxInt // Max int value
	}
	if trimmedEnd <= uint(maxInt) {
		// #nosec G115 - Safe conversion with bounds checking
		entity.EndPos = int(trimmedEnd)
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

// chunkTokens splits tokens into overlapping chunks for processing long texts.
// Truncation is disabled in tokenizer.json so the tokenizer returns all tokens;
// this function handles splitting them into model-sized chunks.
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
