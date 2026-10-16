package pii

import (
	"context"
	"log"
	"strings"
	"time"
)

const (
	mappingCleanupInterval = time.Second
	mappingStatsQueueSize  = 64
	mappingStatsBatchLimit = 1024
	mappingStatsByteLimit  = 128 * 1024
)

// QueueMappingAccesses transfers a bounded copy to one background worker. Counts
// may be dropped under load; request completion never waits for SQLite writes.
func (s *SQLitePIIMappingDB) QueueMappingAccesses(pairs map[string]string) {
	if len(pairs) == 0 || len(pairs) > mappingStatsBatchLimit {
		return
	}
	s.maintenanceMu.Lock()
	defer s.maintenanceMu.Unlock()
	if s.maintenanceStopped || len(s.accessQueue) == cap(s.accessQueue) {
		return
	}
	size := 0
	for original, dummy := range pairs {
		size += len(original) + len(dummy)
		if size > mappingStatsByteLimit {
			return
		}
	}
	batch := make(map[string]string, len(pairs))
	for original, dummy := range pairs {
		batch[strings.Clone(original)] = strings.Clone(dummy)
	}
	select {
	case s.accessQueue <- batch:
	default:
	}
}

func (s *SQLitePIIMappingDB) startMaintenance(interval time.Duration) {
	ctx, cancel := context.WithCancel(context.Background())
	s.maintenanceCancel = cancel
	s.accessQueue = make(chan map[string]string, mappingStatsQueueSize)
	s.maintenanceWG.Add(2)
	go func() {
		defer s.maintenanceWG.Done()
		for {
			select {
			case <-ctx.Done():
				return
			case pairs := <-s.accessQueue:
				if ctx.Err() != nil {
					return
				}
				writeCtx, stop := context.WithTimeout(ctx, mappingStatsTimeout)
				err := s.RecordMappingAccesses(writeCtx, pairs)
				stop()
				if err != nil && ctx.Err() == nil {
					log.Printf("Failed to record mapping accesses: %v", err)
				}
			}
		}
	}()
	go func() {
		defer s.maintenanceWG.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				cleanupCtx, stop := context.WithTimeout(ctx, time.Second)
				err := s.cleanupExpiredMappings(cleanupCtx)
				stop()
				if err != nil && ctx.Err() == nil {
					log.Printf("Failed to clean expired mappings: %v", err)
				}
			}
		}
	}()
}

// Read the persisted policy and remove expired rows atomically. This need not
// acquire mappingMu and therefore cannot delay shutdown behind a request mutex.
// Existing WAL snapshots keep their request-start view after these deletions.
func (s *SQLitePIIMappingDB) cleanupExpiredMappings(ctx context.Context) error {
	_, err := s.maintenanceDB.ExecContext(ctx, `DELETE FROM pii_mappings WHERE EXISTS (
 SELECT 1 FROM pii_mapping_policy WHERE id = 1 AND NOT `+liveMappingAtBoundary+`)`, mappingTimeBoundary())
	return err
}

func (s *SQLitePIIMappingDB) stopMaintenance() {
	s.maintenanceMu.Lock()
	s.maintenanceStopped = true
	s.maintenanceCancel()
	s.maintenanceMu.Unlock()
	s.maintenanceWG.Wait()
	// Discard queued real values on shutdown rather than keeping them in memory.
	for {
		select {
		case <-s.accessQueue:
		default:
			return
		}
	}
}
