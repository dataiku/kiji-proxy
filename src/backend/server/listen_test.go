package server

import (
	"io"
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

			for _, host := range []string{"127.0.0.1", "localhost", "::1"} {
				l, err := listenTCP(host, ":"+port)
				if err == nil {
					_ = l.Close()
					t.Fatalf("listenTCP(%q) succeeded while %s holds port %s", host, h.name, port)
				}
				if !strings.Contains(err.Error(), "address already in use") {
					t.Errorf("listenTCP(%q): unexpected error: %v", host, err)
				}
			}
		})
	}
}

func TestListenTCP_LoopbackListensOnBothAddresses(t *testing.T) {
	if l6, err := net.Listen("tcp6", "[::1]:0"); err != nil {
		t.Skipf("IPv6 loopback unavailable: %v", err)
	} else {
		_ = l6.Close()
	}
	for _, host := range []string{"127.0.0.1", "localhost", "::1"} {
		t.Run(host, func(t *testing.T) {
			port := freePort(t)
			l, err := listenTCP(host, ":"+port)
			if err != nil {
				t.Fatalf("listenTCP on a free port: %v", err)
			}

			go func() {
				for {
					c, err := l.Accept()
					if err != nil {
						return
					}
					_, _ = c.Write([]byte("ok"))
					_ = c.Close()
				}
			}()

			for _, addr := range []string{"127.0.0.1:" + port, "[::1]:" + port} {
				c, err := net.Dial("tcp", addr)
				if err != nil {
					t.Fatalf("dial %s: %v", addr, err)
				}
				got, _ := io.ReadAll(c)
				_ = c.Close()
				if string(got) != "ok" {
					t.Errorf("%s: got %q, want %q", addr, got, "ok")
				}
			}

			// Closing the listener releases both addresses.
			if err := l.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			if _, err := l.Accept(); err == nil {
				t.Error("Accept succeeded after Close")
			}
			for _, network := range []string{"tcp4", "tcp6"} {
				h := net.JoinHostPort(map[string]string{"tcp4": "127.0.0.1", "tcp6": "::1"}[network], port)
				again, err := net.Listen(network, h)
				if err != nil {
					t.Errorf("%s still bound after Close: %v", h, err)
					continue
				}
				_ = again.Close()
			}
		})
	}
}

func TestListenTCP_FailureReleasesTheOtherAddress(t *testing.T) {
	port := freePort(t)
	holder, err := net.Listen("tcp6", net.JoinHostPort("::1", port))
	if err != nil {
		t.Skipf("cannot create holder: %v", err)
	}
	defer func() { _ = holder.Close() }()

	if l, err := listenTCP("127.0.0.1", ":"+port); err == nil {
		_ = l.Close()
		t.Fatal("listenTCP succeeded while [::1] is held")
	}
	l4, err := net.Listen("tcp4", "127.0.0.1:"+port)
	if err != nil {
		t.Fatalf("127.0.0.1 left bound after a failed listenTCP: %v", err)
	}
	_ = l4.Close()
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

func TestListenTCP_NonLoopbackBindsRequestedHost(t *testing.T) {
	l, err := listenTCP("", ":0")
	if err != nil {
		t.Fatalf("listenTCP on all interfaces: %v", err)
	}
	defer func() { _ = l.Close() }()
	if _, ok := l.(*multiListener); ok {
		t.Error("an empty host must not use the loopback pair")
	}
}

func TestIsLoopbackHost(t *testing.T) {
	for host, want := range map[string]bool{
		"127.0.0.1": true, "::1": true, "localhost": true, "LOCALHOST": true,
		"127.0.0.2": false, "": false, "0.0.0.0": false, "::": false, "192.168.1.10": false,
	} {
		if got := isLoopbackHost(host); got != want {
			t.Errorf("isLoopbackHost(%q) = %v, want %v", host, got, want)
		}
	}
}
