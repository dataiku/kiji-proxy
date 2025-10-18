package pii

// DetectorInput represents the input for PII detection
type DetectorInput struct {
	Text string `json:"text"`
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
