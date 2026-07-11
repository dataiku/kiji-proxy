// Package telemetry centralizes all Sentry (error-reporting) wiring for the Go
// backend. It is the single source of truth for the Sentry DSN and options, and
// every other package must go through it rather than importing sentry-go
// directly.
//
// Telemetry is OPT-IN. Nothing is sent unless the operator explicitly enables it
// by setting KIJI_TELEMETRY_ENABLED=true (the desktop app forwards the user's
// Settings choice through this env var when it spawns the backend). When it is
// disabled, Init is a no-op and every other function here short-circuits, so no
// events, traces, or panics ever leave the machine — important for a privacy
// proxy.
package telemetry

import (
	"log"
	"net/http"
	"time"

	"github.com/getsentry/sentry-go"
	sentryhttp "github.com/getsentry/sentry-go/http"
)

// dsn is the Sentry ingest endpoint. It is a public (client-side) DSN, so it is
// safe to embed; it only permits sending events, not reading them. Defined once
// here so the backend never duplicates it (the frontend has its own single copy
// in src/frontend/src/telemetry/sentry.js).
const dsn = "https://d7ad4213601549253c0d313b271f83cf@o4510660510679040.ingest.de.sentry.io/4510660556095568"

// tracesSampleRate keeps performance tracing light: 10% of transactions rather
// than the previous 100%, which was needlessly expensive.
const tracesSampleRate = 0.1

// enabled records whether Init actually brought Sentry up. All exported helpers
// short-circuit on it so callers can invoke them unconditionally.
var enabled bool

// Init brings up Sentry when enable is true; otherwise it does nothing (and
// leaves every other helper a no-op). environment is "production"/"development";
// release is the build version string.
func Init(enable bool, environment, release string) {
	if !enable {
		log.Println("Telemetry disabled (opt-in). Set KIJI_TELEMETRY_ENABLED=true to send crash/error reports to Sentry.")
		return
	}

	if err := sentry.Init(sentry.ClientOptions{
		Dsn:              dsn,
		Environment:      environment,
		Release:          release,
		TracesSampleRate: tracesSampleRate,
	}); err != nil {
		log.Printf("Warning: Failed to initialize Sentry: %v", err)
		return
	}

	enabled = true
	log.Println("Telemetry enabled: Sentry initialized (error and crash reporting active)")
}

// Enabled reports whether telemetry is active.
func Enabled() bool { return enabled }

// CaptureException reports err to Sentry when telemetry is enabled.
func CaptureException(err error) {
	if !enabled {
		return
	}
	sentry.CaptureException(err)
}

// Flush blocks until buffered events are sent or timeout elapses. Safe to call
// when telemetry is disabled (no-op).
func Flush(timeout time.Duration) {
	if !enabled {
		return
	}
	sentry.Flush(timeout)
}

// Recover captures a panic in the current goroutine and forwards it to Sentry,
// then re-panics so the process keeps its original crash behavior. Use as
// `defer telemetry.Recover()` at the top of a goroutine's entry function. It is
// a no-op (and does not swallow the panic) when telemetry is disabled.
func Recover() {
	if r := recover(); r != nil {
		if enabled {
			sentry.CurrentHub().Recover(r)
			sentry.Flush(2 * time.Second)
		}
		panic(r)
	}
}

// HTTPMiddleware wraps an http.Handler so panics in request handlers are
// reported to Sentry before the request is aborted. Repanic is on, so net/http's
// per-connection recovery still runs and the server stays up exactly as before.
// When telemetry is disabled the wrapper is still installed but reports nothing
// (the current hub has no client), so behavior is unchanged.
func HTTPMiddleware(next http.Handler) http.Handler {
	return sentryhttp.New(sentryhttp.Options{Repanic: true}).Handle(next)
}
