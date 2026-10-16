package pii

// DetectorInput represents the input for PII detection
type DetectorInput struct {
	Text string `json:"text"`
	// EntityConfidenceThreshold, when > 0, overrides the detector's default
	// minimum confidence for this call only. It lets each request apply its PII
	// profile's sensitivity without mutating shared detector state.
	EntityConfidenceThreshold float64 `json:"entity_confidence_threshold,omitempty"`
}

// effectiveThreshold returns the per-call threshold when set, else fallback.
func (in DetectorInput) effectiveThreshold(fallback float64) float64 {
	if in.EntityConfidenceThreshold > 0 {
		return in.EntityConfidenceThreshold
	}
	return fallback
}

// DetectorOutput represents the output of PII detection
type DetectorOutput struct {
	Text     string   `json:"text"`
	Entities []Entity `json:"entities"`
}

// Entity represents a detected PII entity
type Entity struct {
	Text       string  `json:"text"`
	Label      string  `json:"label"`
	StartPos   int     `json:"start_pos"`
	EndPos     int     `json:"end_pos"`
	Confidence float64 `json:"confidence"`
}
