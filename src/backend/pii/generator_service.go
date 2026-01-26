package pii

import (
	"math/rand"
	"time"

	piiGenerators "github.com/hannes/yaak-private/src/backend/pii/generators"
)

// GeneratorService handles PII replacement generation
type GeneratorService struct {
	rng *rand.Rand
}

// NewGeneratorService creates a new generator service
func NewGeneratorService() *GeneratorService {
	// #nosec G404 - Using math/rand for deterministic PII generation, not security-critical
	return &GeneratorService{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// GenerateReplacement generates a replacement for the given PII label and original text
func (s *GeneratorService) GenerateReplacement(label, originalText string) string {
	generator := s.getGeneratorForLabel(label)
	return generator(originalText)
}

// getGeneratorForLabel returns the appropriate generator function for the given label
func (s *GeneratorService) getGeneratorForLabel(label string) func(string) string {
	generators := map[string]func(string) string{
		"EMAIL":            func(original string) string { return piiGenerators.EmailGenerator(s.rng, original) },
		"SOCIALNUM":        func(original string) string { return piiGenerators.SSNGenerator(s.rng, original) },
		"SSN":              func(original string) string { return piiGenerators.SSNGenerator(s.rng, original) },
		"TELEPHONENUM":     func(original string) string { return piiGenerators.PhoneGenerator(s.rng, original) },
		"PHONENUMBER":      func(original string) string { return piiGenerators.PhoneGenerator(s.rng, original) },
		"CREDITCARDNUMBER": func(original string) string { return piiGenerators.CreditCardGenerator(s.rng, original) },
		"USERNAME":         func(original string) string { return piiGenerators.UsernameGenerator(s.rng, original) },
		"DATEOFBIRTH":      func(original string) string { return piiGenerators.DateOfBirthGenerator(s.rng, original) },
		"ZIPCODE":          func(original string) string { return piiGenerators.ZipCodeGenerator(s.rng, original) },
		"ZIP":              func(original string) string { return piiGenerators.ZipCodeGenerator(s.rng, original) },
		"ACCOUNTNUM":       func(original string) string { return piiGenerators.AccountNumGenerator(s.rng, original) },
		"IDCARDNUM":        func(original string) string { return piiGenerators.IDCardNumGenerator(s.rng, original) },
		"DRIVERLICENSENUM": func(original string) string { return piiGenerators.DriverLicenseNumGenerator(s.rng, original) },
		"TAXNUM":           func(original string) string { return piiGenerators.TaxNumGenerator(s.rng, original) },
		"CITY":             func(original string) string { return piiGenerators.CityGenerator(s.rng, original) },
		"STREET":           func(original string) string { return piiGenerators.StreetGenerator(s.rng, original) },
		"BUILDINGNUM":      func(original string) string { return piiGenerators.BuildingNumGenerator(s.rng, original) },
		"FIRSTNAME":        func(original string) string { return piiGenerators.FirstNameGenerator(s.rng, original) },
		"SURNAME":          func(original string) string { return piiGenerators.SurnameGenerator(s.rng, original) },
		"URL":              func(original string) string { return piiGenerators.UrlGenerator(s.rng, original) },
		"COMPANYNAME":      func(original string) string { return piiGenerators.CompanyNameGenerator(s.rng, original) },
		"STATE":            func(original string) string { return piiGenerators.StateGenerator(s.rng, original) },
		"COUNTRY":          func(original string) string { return piiGenerators.CountryGenerator(s.rng, original) },
		"PASSPORTID":       func(original string) string { return piiGenerators.PassportIdGenerator(s.rng, original) },
		"NATIONALID":       func(original string) string { return piiGenerators.NationalIdGenerator(s.rng, original) },
		"LICENSEPLATENUM":  func(original string) string { return piiGenerators.LicensePlateNumGenerator(s.rng, original) },
		"PASSWORD":         func(original string) string { return piiGenerators.PasswordGenerator(s.rng, original) },
		"IBAN":             func(original string) string { return piiGenerators.IbanGenerator(s.rng, original) },
		"AGE":              func(original string) string { return piiGenerators.AgeGenerator(s.rng, original) },
		"SECURITYTOKEN":    func(original string) string { return piiGenerators.SecurityTokenGenerator(s.rng, original) },
	}

	if generator, exists := generators[label]; exists {
		return generator
	}

	// Return generic generator for unknown labels
	return func(original string) string { return piiGenerators.GenericGenerator(s.rng, original) }
}
