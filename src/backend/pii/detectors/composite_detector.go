package pii

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
)

// CompositeDetector runs several detectors over the same input and merges their
// results into a single set of entities. It implements the Detector interface, so
// it can be used anywhere a single detector is expected (e.g. as the detector a
// ModelManager hands out). Detectors run in the order supplied; overlapping
// matches are reconciled by mergeChunkEntities, which prefers higher-confidence
// spans.
type CompositeDetector struct {
	detectors []Detector
}

// NewCompositeDetector builds a CompositeDetector over the supplied detectors.
func NewCompositeDetector(detectors ...Detector) *CompositeDetector {
	return &CompositeDetector{detectors: detectors}
}

// GetName returns the detector's identifier.
func (c *CompositeDetector) GetName() string {
	return "composite_detector"
}

// SetEntityConfidenceThreshold forwards the threshold to every child detector.
func (c *CompositeDetector) SetEntityConfidenceThreshold(threshold float64) {
	for _, d := range c.detectors {
		d.SetEntityConfidenceThreshold(threshold)
	}
}

// EntityTypes returns the deduplicated, sorted union of every child detector's
// entity types.
func (c *CompositeDetector) EntityTypes() []string {
	seen := make(map[string]struct{})
	for _, d := range c.detectors {
		for _, t := range d.EntityTypes() {
			seen[t] = struct{}{}
		}
	}

	types := make([]string, 0, len(seen))
	for t := range seen {
		types = append(types, t)
	}
	sort.Strings(types)
	return types
}

// Detect runs every child detector and returns the merged set of entities. Each
// detector's entities are treated as one group and reconciled by
// mergeChunkEntities, which resolves overlaps in favor of higher confidence.
//
// A single detector failing does not fail the whole detection: its error is
// logged and the entities from the detectors that succeeded are still returned.
// Failing open on one detector would leak PII the others can still catch. Detect
// only returns an error when every detector failed.
func (c *CompositeDetector) Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error) {
	groups := make([][]Entity, 0, len(c.detectors))
	var errs []error

	for _, d := range c.detectors {
		// Respect cancellation between detectors so large inputs stay responsive.
		select {
		case <-ctx.Done():
			return DetectorOutput{}, ctx.Err()
		default:
		}

		out, err := d.Detect(ctx, input)
		if err != nil {
			errs = append(errs, fmt.Errorf("detector %q: %w", d.GetName(), err))
			continue
		}
		groups = append(groups, out.Entities)
	}

	if len(groups) == 0 && len(errs) > 0 {
		return DetectorOutput{}, errors.Join(errs...)
	}
	if len(errs) > 0 {
		log.Printf("[CompositeDetector] %d detector(s) failed, continuing with the rest: %v", len(errs), errors.Join(errs...))
	}

	return DetectorOutput{
		Text:     input.Text,
		Entities: mergeChunkEntities(groups, input.Text),
	}, nil
}

// Close closes every child detector, attempting all of them and returning the
// joined error if any failed.
func (c *CompositeDetector) Close() error {
	var errs []error
	for _, d := range c.detectors {
		if err := d.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
