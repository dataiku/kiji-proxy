package server

import (
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// listenTCP binds host:port. A loopback host ("127.0.0.1", "::1" or
// "localhost") always means both loopback addresses, 127.0.0.1 and [::1], so
// clients reach Kiji whichever one "localhost" resolves to. If IPv6 is
// unavailable only 127.0.0.1 is bound.
//
// Startup fails with "address already in use" when another process holds
// either address. Before binding, it also checks the addresses a plain bind
// would silently share (on macOS the wildcards 0.0.0.0 and [::]): Go sets
// SO_REUSEADDR, so without the check two processes would share the port and
// clients would reach whichever one the OS picks.
func listenTCP(host, port string) (net.Listener, error) {
	port = strings.TrimPrefix(port, ":")
	if !isLoopbackHost(host) {
		return net.Listen("tcp", config.JoinListenAddr(host, port))
	}
	if port != "0" {
		for _, probe := range loopbackCollisionProbes {
			if err := probe(port); err != nil {
				return nil, fmt.Errorf("port %s is already in use by another process: %w", port, err)
			}
		}
	}

	v4, err := net.Listen("tcp4", net.JoinHostPort("127.0.0.1", port))
	if err != nil {
		return nil, err
	}
	// With port 0 the OS picked a port for v4; use the same one for v6.
	v6Port := strconv.Itoa(v4.Addr().(*net.TCPAddr).Port)
	v6, err := net.Listen("tcp6", net.JoinHostPort("::1", v6Port))
	if err != nil {
		if errors.Is(err, syscall.EADDRINUSE) {
			_ = v4.Close()
			return nil, err
		}
		// IPv6 is unavailable or has no loopback address: serve IPv4 only.
		return v4, nil
	}
	return newMultiListener(v4, v6), nil
}

// multiListener accepts connections from several listeners and closes them
// together. Addr reports the first listener's address.
type multiListener struct {
	listeners []net.Listener
	accepted  chan acceptResult
	done      chan struct{}
	closeOnce sync.Once
}

type acceptResult struct {
	conn net.Conn
	err  error
}

func newMultiListener(listeners ...net.Listener) *multiListener {
	m := &multiListener{
		listeners: listeners,
		accepted:  make(chan acceptResult),
		done:      make(chan struct{}),
	}
	for _, l := range listeners {
		go m.acceptLoop(l)
	}
	return m
}

func (m *multiListener) acceptLoop(l net.Listener) {
	for {
		conn, err := l.Accept()
		select {
		case m.accepted <- acceptResult{conn, err}:
		case <-m.done:
			if conn != nil {
				_ = conn.Close()
			}
			return
		}
		if err != nil && errors.Is(err, net.ErrClosed) {
			return
		}
	}
}

func (m *multiListener) Accept() (net.Conn, error) {
	select {
	case r := <-m.accepted:
		return r.conn, r.err
	case <-m.done:
		return nil, net.ErrClosed
	}
}

func (m *multiListener) Close() error {
	var err error
	m.closeOnce.Do(func() {
		close(m.done)
		for _, l := range m.listeners {
			err = errors.Join(err, l.Close())
		}
	})
	return err
}

func (m *multiListener) Addr() net.Addr {
	return m.listeners[0].Addr()
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	// Only the canonical addresses: any other 127.x.x.x host is bound as given.
	ip := net.ParseIP(host)
	return ip != nil && (ip.Equal(net.IPv4(127, 0, 0, 1)) || ip.Equal(net.IPv6loopback))
}
