package pii

import (
	"strings"
	"testing"
)

// generatorLabels is every label GenerateReplacement routes to a dedicated
// generator, plus an unknown label to cover the generic fallback path.
var generatorLabels = []string{
	labelSurname, labelFirstName, labelBuildingNum, labelDateOfBirth,
	labelEmail, labelPhoneNumber, labelCity, labelURL, labelCompanyName,
	labelState, labelZip, labelStreet, labelCountry, labelSSN,
	labelDriverLicenseNum, labelPassportID, labelNationalID, labelIDCardNum,
	labelTaxNum, labelLicensePlateNum, labelPassword, labelIBAN, labelAge,
	labelSecurityToken, labelCreditCardNumber, labelUsername,
	"UNKNOWNLABEL",
}

// A replacement equal to the original would leak the PII it is supposed to
// mask, so GenerateReplacement guarantees inequality. Feeding a previous
// output back in as the original is the most collision-prone input possible,
// and must still always produce something different.
func TestGenerateReplacementNeverReturnsOriginal(t *testing.T) {
	service := NewGeneratorService()
	for _, label := range generatorLabels {
		original := service.GenerateReplacement(label, "")
		for i := 0; i < 100; i++ {
			if replacement := service.GenerateReplacement(label, original); replacement == original {
				t.Fatalf("GenerateReplacement(%q) returned the original %q", label, original)
			}
		}
	}
}

func TestGenerateReplacementUnknownLabelUsesPlaceholder(t *testing.T) {
	service := NewGeneratorService()
	replacement := service.GenerateReplacement("SOMETHINGELSE", "raw value")
	if !strings.HasPrefix(replacement, "[REDACTED_SOMETHINGELSE_") {
		t.Errorf("expected generic placeholder for unknown label, got %q", replacement)
	}
}
