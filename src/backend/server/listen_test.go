package server

import (
	"net"
	"strconv"
	"strings"
	"testing"
)

// freePort returns a port that is currently free on every address.
func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}
	port := strconv.Itoa(l.Addr().(*net.TCPAddr).Port)
	_ = l.Close()
	return port
}

func TestListenTCP_LoopbackFailsWhenPortIsHeld(t *testing.T) {
	holders := []struct{ name, network, host string }{
		{"127.0.0.1", "tcp4", "127.0.0.1"},
		{"[::1]", "tcp6", "::1"},
		{"0.0.0.0", "tcp4", "0.0.0.0"},
		{"[::]", "tcp6", "::"},
		{"* (dual stack)", "tcp", ""},
	}
	for _, h := range holders {
		t.Run(h.name, func(t *testing.T) {
			port := freePort(t)
			holder, err := net.Listen(h.network, net.JoinHostPort(h.host, port))
			if err != nil {
				t.Skipf("cannot create holder: %v", err) // e.g. no IPv6
			}
			defer func() { _ = holder.Close() }()

			l, err := listenTCP("127.0.0.1", ":"+port)
			if err == nil {
				_ = l.Close()
				t.Fatalf("listenTCP succeeded while %s holds port %s", h.name, port)
			}
			if !strings.Contains(err.Error(), "address already in use") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestListenTCP_LoopbackFreePort(t *testing.T) {
	port := freePort(t)
	l, err := listenTCP("127.0.0.1", ":"+port)
	if err != nil {
		t.Fatalf("listenTCP on a free port: %v", err)
	}
	defer func() { _ = l.Close() }()
	if got := l.Addr().String(); got != "127.0.0.1:"+port {
		t.Errorf("bound %s, want 127.0.0.1:%s", got, port)
	}

	// Port 0 skips the probes and still works.
	l0, err := listenTCP("127.0.0.1", ":0")
	if err != nil {
		t.Fatalf("listenTCP on port 0: %v", err)
	}
	_ = l0.Close()
}

func TestIsLoopbackHost(t *testing.T) {
	for host, want := range map[string]bool{
		"127.0.0.1": true, "::1": true, "localhost": true, "LOCALHOST": true,
		"": false, "0.0.0.0": false, "::": false, "192.168.1.10": false,
	} {
		if got := isLoopbackHost(host); got != want {
			t.Errorf("isLoopbackHost(%q) = %v, want %v", host, got, want)
		}
	}
}
