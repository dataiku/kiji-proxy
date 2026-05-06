// Package pdf provides experimental PDF redaction that runs each page's
// extracted text through the Kiji PII masking pipeline and rewrites the
// page content with the masked text rendered in a built-in standard font
// (Helvetica).
//
// This is wholesale-replacement masking: the entire page content stream is
// discarded and re-emitted as plain text. The original PII bytes do not
// appear anywhere in the output file. Original layout, embedded fonts,
// images, tables, and form structure are NOT preserved — the output is a
// flat plain-text rendering of the masked content. Use this when reliable
// PII removal matters more than visual fidelity.
//
// unipdf is a commercial library; set UNIDOC_LICENSE_API_KEY in the
// environment before invoking these functions.
package pdf

import (
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/unidoc/unipdf/v4/common/license"
	"github.com/unidoc/unipdf/v4/core"
	"github.com/unidoc/unipdf/v4/extractor"
	"github.com/unidoc/unipdf/v4/model"

	pii "github.com/hannes/kiji-private/src/backend/pii"
)

// Masker is the subset of pii.MaskingService used by the redaction pipeline.
// Defining it as an interface lets callers swap in alternative implementations
// (e.g. an HTTP client that calls /api/pii/check).
type Masker interface {
	MaskText(text string, logPrefix string) pii.MaskedResult
}

// licenseOnce gates the unipdf license registration so we read the env exactly
// once, lazily — at the first MaskPDF call. We can't use a package init() for
// this because it runs before main() loads the .env file, which would leave
// UNIDOC_LICENSE_API_KEY unset when SetMeteredKey is called.
var licenseOnce sync.Once

func ensureLicense() {
	licenseOnce.Do(func() {
		key := os.Getenv("UNIDOC_LICENSE_API_KEY")
		if key == "" {
			return
		}
		if err := license.SetMeteredKey(key); err != nil {
			log.Printf("[PDF] failed to set unipdf metered license key: %v", err)
		}
	})
}

const (
	maskFontSize        = 12.0
	maskFontLeading     = maskFontSize * 1.2
	maskMargin          = 50.0
	maskFontResourceKey = "F_KIJI_HELV"
)

// MaskPDF reads a PDF from in, runs each page's extracted text through masker
// as a single block, and writes a new PDF to out where every page's content
// stream has been replaced with the masked text rendered in Helvetica.
func MaskPDF(in io.ReadSeeker, out io.Writer, masker Masker) error {
	ensureLicense()
	reader, err := model.NewPdfReader(in)
	if err != nil {
		return fmt.Errorf("read PDF: %w", err)
	}
	numPages, err := reader.GetNumPages()
	if err != nil {
		return fmt.Errorf("get page count: %w", err)
	}

	writer := model.NewPdfWriter()
	for pageNum := 1; pageNum <= numPages; pageNum++ {
		page, err := reader.GetPage(pageNum)
		if err != nil {
			return fmt.Errorf("get page %d: %w", pageNum, err)
		}
		if err := rewritePage(page, masker, pageNum); err != nil {
			return fmt.Errorf("rewrite page %d: %w", pageNum, err)
		}
		if err := writer.AddPage(page); err != nil {
			return fmt.Errorf("add page %d: %w", pageNum, err)
		}
	}

	if err := writer.Write(out); err != nil {
		return fmt.Errorf("write PDF: %w", err)
	}
	return nil
}

// MaskPDFFile is a convenience wrapper around MaskPDF that operates on
// filesystem paths.
func MaskPDFFile(inPath, outPath string, masker Masker) error {
	in, err := os.Open(inPath) // #nosec G304 - inPath is supplied by trusted caller
	if err != nil {
		return fmt.Errorf("open %q: %w", inPath, err)
	}
	defer func() { _ = in.Close() }()

	out, err := os.Create(outPath) // #nosec G304 - outPath is supplied by trusted caller
	if err != nil {
		return fmt.Errorf("create %q: %w", outPath, err)
	}
	defer func() { _ = out.Close() }()

	return MaskPDF(in, out, masker)
}

func rewritePage(page *model.PdfPage, masker Masker, pageNum int) error {
	ex, err := extractor.New(page)
	if err != nil {
		return fmt.Errorf("extractor: %w", err)
	}
	pageText, _, _, err := ex.ExtractPageText()
	if err != nil {
		return fmt.Errorf("extract text: %w", err)
	}
	text := pageText.Text()
	if text == "" {
		log.Printf("[PDF] p%d: no text on page; clearing content stream", pageNum)
		return page.SetContentStreams([]string{""}, core.NewFlateEncoder())
	}

	result := masker.MaskText(text, fmt.Sprintf("[PDF p%d]", pageNum))
	log.Printf("[PDF] p%d: %d entities detected", pageNum, len(result.Entities))
	for i, e := range result.Entities {
		var surrogate string
		if i < len(result.EntityReplacements) {
			surrogate = result.EntityReplacements[i]
		}
		log.Printf("[PDF] p%d: entity %s %q -> %q (offset %d-%d, conf %.2f)",
			pageNum, e.Label, e.Text, surrogate, e.StartPos, e.EndPos, e.Confidence)
	}

	mediaBox, err := page.GetMediaBox()
	if err != nil {
		return fmt.Errorf("get media box: %w", err)
	}

	helv, err := model.NewStandard14Font(model.HelveticaName)
	if err != nil {
		return fmt.Errorf("load Helvetica: %w", err)
	}
	if page.Resources == nil {
		page.Resources = model.NewPdfPageResources()
	}
	if err := page.Resources.SetFontByName(core.PdfObjectName(maskFontResourceKey), helv.ToPdfObject()); err != nil {
		return fmt.Errorf("install replacement font: %w", err)
	}

	stream, err := buildMaskedContentStream(result.MaskedText, mediaBox, helv)
	if err != nil {
		return fmt.Errorf("build content stream: %w", err)
	}

	return page.SetContentStreams([]string{stream}, core.NewFlateEncoder())
}

// buildMaskedContentStream emits a fresh PDF content stream that draws the
// masked text using Helvetica at the top of the page, line by line. The text
// is encoded through Helvetica's own encoder (WinAnsi by default) and emitted
// as PDF hex strings so multi-byte UTF-8 input doesn't get interpreted as
// individual WinAnsi bytes by the reader.
func buildMaskedContentStream(masked string, mediaBox *model.PdfRectangle, font *model.PdfFont) (string, error) {
	encoder := font.Encoder()
	if encoder == nil {
		return "", fmt.Errorf("Helvetica encoder unavailable")
	}

	pageWidth := mediaBox.Urx - mediaBox.Llx
	startX := mediaBox.Llx + maskMargin
	startY := mediaBox.Ury - maskMargin - maskFontSize

	// Helvetica @ 12pt averages ~6pt per character. A safety factor of 0.55
	// keeps the right margin clean across mixed-width content.
	avgCharWidth := maskFontSize * 0.55
	usableWidth := pageWidth - 2*maskMargin
	maxChars := max(int(usableWidth/avgCharWidth), 10)

	lines := wrapMaskedText(masked, maxChars)

	var b strings.Builder
	fmt.Fprintf(&b, "BT\n/%s %.2f Tf\n%.2f TL\n%.2f %.2f Td\n",
		maskFontResourceKey, maskFontSize, maskFontLeading, startX, startY)
	for _, line := range lines {
		encoded := encoder.Encode(line)
		fmt.Fprintf(&b, "<%s> Tj T*\n", hex.EncodeToString(encoded))
	}
	b.WriteString("ET\n")
	return b.String(), nil
}

// wrapMaskedText splits masked into lines, breaking on existing newlines and
// further word-wrapping any line that exceeds maxChars.
func wrapMaskedText(masked string, maxChars int) []string {
	var lines []string
	for paragraph := range strings.SplitSeq(masked, "\n") {
		if paragraph == "" {
			lines = append(lines, "")
			continue
		}
		if len(paragraph) <= maxChars {
			lines = append(lines, paragraph)
			continue
		}
		words := strings.Fields(paragraph)
		var current strings.Builder
		for _, word := range words {
			switch {
			case current.Len() == 0:
				current.WriteString(word)
			case current.Len()+1+len(word) <= maxChars:
				current.WriteByte(' ')
				current.WriteString(word)
			default:
				lines = append(lines, current.String())
				current.Reset()
				current.WriteString(word)
			}
		}
		if current.Len() > 0 {
			lines = append(lines, current.String())
		}
	}
	return lines
}
