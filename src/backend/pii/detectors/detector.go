package pii

import (
	"context"
)

type Detector interface {
	GetName() string
	Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error)
	Close() error
}

func CloseDetector(detector Detector) error {
	return detector.Close()
}
