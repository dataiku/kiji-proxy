package pii

import (
	"context"
)

type Detector interface {
	GetName() string
	Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error)
	SetEntityConfidenceThreshold(threshold float64)
	// EntityTypes returns the deduplicated set of base entity labels this model
	// can emit (BIO prefixes stripped, non-entity sentinels O/IGNORE excluded),
	// sorted alphabetically. This is the maximum set of selectable entities.
	EntityTypes() []string
	Close() error
}

func CloseDetector(detector Detector) error {
	return detector.Close()
}
