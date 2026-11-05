package proxy

import "testing"

func TestSystemProxyManager_DisableWithoutEnableIsNoop(t *testing.T) {
	// An empty PATH makes any networksetup call fail instead of changing the
	// real system proxy settings, so a regression shows up as an error here.
	t.Setenv("PATH", "")

	m := NewSystemProxyManager("http://127.0.0.1:9090/proxy.pac")
	if m.Enabled() {
		t.Fatal("new manager reports Enabled")
	}
	if err := m.Disable(); err != nil {
		t.Fatalf("Disable without Enable ran networksetup: %v", err)
	}
}
