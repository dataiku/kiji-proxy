//go:build !darwin

package server

// Elsewhere the 127.0.0.1 and [::1] binds themselves fail against wildcard
// holders and holders on either loopback address, so no extra probes are needed.
var loopbackCollisionProbes []func(port string) error
