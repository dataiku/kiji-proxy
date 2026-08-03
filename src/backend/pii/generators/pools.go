package pii

import (
	"bufio"
	"embed"
	"fmt"
	"io"
	"strings"
)

// poolData holds the newline-delimited word pools the generators draw from.
// Keeping them as embedded data files rather than inline slices lets the pools
// grow (and gain locale-aware variants later) without touching Go code, while
// the binary stays self-contained.
//
//go:embed data/*.txt
var poolData embed.FS

// parsePool reads a newline-delimited pool from r. Surrounding whitespace is
// trimmed, and blank lines and comment lines (those beginning with '#') are
// skipped so the data files can carry headers and section labels.
func parsePool(r io.Reader) []string {
	var entries []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		entries = append(entries, line)
	}
	if err := scanner.Err(); err != nil {
		panic(fmt.Sprintf("pii generators: reading pool: %v", err))
	}
	return entries
}

// loadPool loads a pool file from the embedded data/ directory. It panics on a
// missing, unreadable, or empty file: the files are embedded at compile time,
// so any of those is a build or packaging error that should fail loudly at
// startup rather than leave a generator with an empty pool.
func loadPool(name string) []string {
	f, err := poolData.Open("data/" + name)
	if err != nil {
		panic(fmt.Sprintf("pii generators: cannot open embedded pool %q: %v", name, err))
	}
	defer f.Close()

	entries := parsePool(f)
	if len(entries) == 0 {
		panic(fmt.Sprintf("pii generators: embedded pool %q is empty", name))
	}
	return entries
}

// Word pools loaded once at package initialization from the embedded data
// files. The generators reference these directly.
var (
	emailFirstNames = loadPool("email_firstnames.txt")
	emailSurnames   = loadPool("email_surnames.txt")
	firstNames      = loadPool("firstnames.txt")
	surnames        = loadPool("surnames.txt")
	cities          = loadPool("cities.txt")
	streets         = loadPool("streets.txt")
	states          = loadPool("states.txt")
	countries       = loadPool("countries.txt")
	companyPrefixes = loadPool("company_prefixes.txt")
	companySuffixes = loadPool("company_suffixes.txt")
)
