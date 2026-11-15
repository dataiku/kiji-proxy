package server

import (
	"fmt"
	"strconv"
	"syscall"
)

// On macOS a 127.0.0.1 bind also succeeds while another process holds the
// port on 0.0.0.0 or [::] (including dual-stack "*"), so probe those too.
// (On Linux the 127.0.0.1 bind itself fails in that case.)
var loopbackCollisionProbes = []func(port string) error{
	probeIPv6Loopback,
	func(port string) error { return probeWildcard(syscall.AF_INET, port) },
	func(port string) error { return probeWildcard(syscall.AF_INET6, port) },
}

// probeWildcard fails when another process holds the wildcard address of the
// given family on port. It only binds, never listens, so it never accepts a
// connection on a non-loopback interface (and never triggers the macOS
// firewall prompt). SO_REUSEADDR matches Go's net.Listen, so holders on other
// specific addresses (a LAN IP, Docker's bridge) do not count.
func probeWildcard(family int, port string) error {
	p, err := strconv.Atoi(port)
	if err != nil {
		return nil // not a number: let net.Listen report it
	}
	fd, err := syscall.Socket(family, syscall.SOCK_STREAM, 0)
	if err != nil {
		return nil // family unavailable, e.g. IPv6 disabled
	}
	defer func() { _ = syscall.Close(fd) }()
	if err := syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1); err != nil {
		return nil
	}
	var sa syscall.Sockaddr
	network, addr := "tcp4", fmt.Sprintf("0.0.0.0:%d", p)
	if family == syscall.AF_INET6 {
		_ = syscall.SetsockoptInt(fd, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 1)
		sa = &syscall.SockaddrInet6{Port: p}
		network, addr = "tcp6", fmt.Sprintf("[::]:%d", p)
	} else {
		sa = &syscall.SockaddrInet4{Port: p}
	}
	if err := syscall.Bind(fd, sa); err == syscall.EADDRINUSE {
		// Same shape as net.Listen errors, which the desktop app parses.
		return fmt.Errorf("listen %s %s: bind: %w", network, addr, err)
	}
	return nil
}
