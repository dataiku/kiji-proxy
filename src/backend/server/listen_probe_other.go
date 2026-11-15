//go:build !darwin

package server

// Elsewhere a 127.0.0.1 bind already fails against wildcard holders; only a
// holder on [::1] (which "localhost" may resolve to) needs a separate check.
var loopbackCollisionProbes = []func(port string) error{
	probeIPv6Loopback,
}
