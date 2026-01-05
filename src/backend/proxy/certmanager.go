package proxy

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"log"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// CertManager handles certificate generation and caching for MITM proxy
type CertManager struct {
	caCert     *x509.Certificate
	caKey      *rsa.PrivateKey
	certCache  map[string]*tls.Certificate
	cacheMutex sync.RWMutex
	caPath     string
	keyPath    string
}

// NewCertManager creates a new certificate manager
func NewCertManager(caPath, keyPath string) (*CertManager, error) {
	cm := &CertManager{
		certCache: make(map[string]*tls.Certificate),
		caPath:    caPath,
		keyPath:   keyPath,
	}

	// Try to load existing CA certificate
	if err := cm.loadCA(); err != nil {
		log.Printf("Failed to load CA certificate, generating new one: %v", err)
		if err := cm.generateCA(); err != nil {
			return nil, fmt.Errorf("failed to generate CA certificate: %w", err)
		}
	}

	return cm, nil
}

// loadCA loads the CA certificate and key from disk
func (cm *CertManager) loadCA() error {
	// Load certificate
	certPEM, err := os.ReadFile(cm.caPath)
	if err != nil {
		return err
	}

	// Load private key
	keyPEM, err := os.ReadFile(cm.keyPath)
	if err != nil {
		return err
	}

	// Parse certificate
	block, _ := pem.Decode(certPEM)
	if block == nil {
		return fmt.Errorf("failed to decode CA certificate PEM")
	}
	cm.caCert, err = x509.ParseCertificate(block.Bytes)
	if err != nil {
		return fmt.Errorf("failed to parse CA certificate: %w", err)
	}

	// Parse private key
	keyBlock, _ := pem.Decode(keyPEM)
	if keyBlock == nil {
		return fmt.Errorf("failed to decode CA key PEM")
	}
	cm.caKey, err = x509.ParsePKCS1PrivateKey(keyBlock.Bytes)
	if err != nil {
		return fmt.Errorf("failed to parse CA key: %w", err)
	}

	log.Printf("Loaded CA certificate from %s", cm.caPath)
	return nil
}

// generateCA generates a new CA certificate and key
func (cm *CertManager) generateCA() error {
	// Generate private key
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return fmt.Errorf("failed to generate CA key: %w", err)
	}

	// Create certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization:  []string{"Yaak Proxy CA"},
			Country:       []string{"US"},
			Province:      []string{""},
			Locality:      []string{""},
			StreetAddress: []string{""},
			PostalCode:    []string{""},
			CommonName:    "Yaak Proxy CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(10, 0, 0), // 10 years
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
	}

	// Create self-signed certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
	if err != nil {
		return fmt.Errorf("failed to create CA certificate: %w", err)
	}

	cm.caCert, err = x509.ParseCertificate(certDER)
	if err != nil {
		return fmt.Errorf("failed to parse generated CA certificate: %w", err)
	}
	cm.caKey = key

	// Save to disk
	if err := cm.saveCA(); err != nil {
		return fmt.Errorf("failed to save CA certificate: %w", err)
	}

	log.Printf("Generated new CA certificate and saved to %s", cm.caPath)
	return nil
}

// saveCA saves the CA certificate and key to disk
func (cm *CertManager) saveCA() error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(cm.caPath), 0755); err != nil {
		return fmt.Errorf("failed to create certificate directory: %w", err)
	}

	// Save certificate
	certPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cm.caCert.Raw,
	})
	if err := os.WriteFile(cm.caPath, certPEM, 0644); err != nil {
		return fmt.Errorf("failed to write CA certificate: %w", err)
	}

	// Save private key
	keyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(cm.caKey),
	})
	if err := os.WriteFile(cm.keyPath, keyPEM, 0600); err != nil {
		return fmt.Errorf("failed to write CA key: %w", err)
	}

	return nil
}

// GetCACert returns the CA certificate in PEM format
func (cm *CertManager) GetCACert() []byte {
	return pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cm.caCert.Raw,
	})
}

// GetCert generates or retrieves a cached certificate for the given hostname
func (cm *CertManager) GetCert(hostname string) (*tls.Certificate, error) {
	// Check cache first
	cm.cacheMutex.RLock()
	if cert, ok := cm.certCache[hostname]; ok {
		cm.cacheMutex.RUnlock()
		log.Printf("[CertManager] Using cached certificate for %s", hostname)
		return cert, nil
	}
	cm.cacheMutex.RUnlock()

	// Generate new certificate
	log.Printf("[CertManager] Generating new certificate for %s", hostname)
	cert, err := cm.generateCert(hostname)
	if err != nil {
		log.Printf("[CertManager] ❌ Failed to generate certificate for %s: %v", hostname, err)
		return nil, err
	}

	// Cache it
	cm.cacheMutex.Lock()
	cm.certCache[hostname] = cert
	cm.cacheMutex.Unlock()

	log.Printf("[CertManager] ✓ Successfully generated and cached certificate for %s", hostname)
	return cert, nil
}

// generateCert generates a certificate for the given hostname
func (cm *CertManager) generateCert(hostname string) (*tls.Certificate, error) {
	// Generate private key
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}

	// Parse hostname to extract domain and IP
	host, _, err := net.SplitHostPort(hostname)
	if err != nil {
		host = hostname
	}

	// Create certificate template
	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("failed to generate serial number: %w", err)
	}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"Yaak Proxy"},
			Country:      []string{"US"},
			CommonName:   host,
			SerialNumber: serialNumber.String(),
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(1, 0, 0), // 1 year
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  false,
	}

	// Add hostname to SAN
	if ip := net.ParseIP(host); ip != nil {
		template.IPAddresses = []net.IP{ip}
		log.Printf("[CertManager] Adding IP address to certificate: %s", ip)
	} else {
		template.DNSNames = []string{host}
		// Also add wildcard for subdomains
		if host != "" {
			template.DNSNames = append(template.DNSNames, "*."+host)
		}
		log.Printf("[CertManager] Adding DNS names to certificate: %v", template.DNSNames)
	}

	// Sign certificate with CA
	certDER, err := x509.CreateCertificate(rand.Reader, &template, cm.caCert, &key.PublicKey, cm.caKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	// Include both the leaf certificate and the CA certificate in the chain
	// This helps clients validate the certificate chain
	cert := &tls.Certificate{
		Certificate: [][]byte{certDER, cm.caCert.Raw},
		PrivateKey:  key,
	}

	log.Printf("[CertManager] Certificate created with %d certificates in chain (leaf + CA)", len(cert.Certificate))
	return cert, nil
}
