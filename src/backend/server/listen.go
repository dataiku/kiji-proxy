package server

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"syscall"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// listenTCP binds host:port. For a loopback host it first checks that no
// other process holds the same port on an address that also receives
// loopback or "localhost" traffic: [::1], and on macOS the wildcards 0.0.0.0
// and [::]. A plain 127.0.0.1 bind succeeds next to those holders (Go sets
// SO_REUSEADDR), so without the check two processes would share the port and
// clients would reach whichever one the OS picks.
func listenTCP(host, port string) (net.Listener, error) {
	port = strings.TrimPrefix(port, ":")
	if isLoopbackHost(host) && port != "0" {
		for _, probe := range loopbackCollisionProbes {
			if err := probe(port); err != nil {
				return nil, fmt.Errorf("port %s is already in use by another process: %w", port, err)
			}
		}
	}
	return net.Listen("tcp", config.JoinListenAddr(host, port))
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// probeIPv6Loopback fails when another process listens on [::1]:port.
// Other errors (e.g. IPv6 disabled) are not collisions and are ignored.
func probeIPv6Loopback(port string) error {
	l, err := net.Listen("tcp6", net.JoinHostPort("::1", port))
	if err != nil {
		if errors.Is(err, syscall.EADDRINUSE) {
			return err
		}
		return nil
	}
	return l.Close()
}
