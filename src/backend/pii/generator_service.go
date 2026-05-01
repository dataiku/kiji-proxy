package pii

import (
	"math/rand"
	"time"

	piiGenerators "github.com/hannes/kiji-private/src/backend/pii/generators"
)

// PII entity labels recognized by the generator service.
const (
	labelSurname          = "SURNAME"
	labelFirstName        = "FIRSTNAME"
	labelBuildingNum      = "BUILDINGNUM"
	labelDateOfBirth      = "DATEOFBIRTH"
	labelEmail            = "EMAIL"
	labelPhoneNumber      = "PHONENUMBER"
	labelCity             = "CITY"
	labelURL              = "URL"
	labelCompanyName      = "COMPANYNAME"
	labelState            = "STATE"
	labelZip              = "ZIP"
	labelStreet           = "STREET"
	labelCountry          = "COUNTRY"
	labelSSN              = "SSN"
	labelDriverLicenseNum = "DRIVERLICENSENUM"
	labelPassportID       = "PASSPORTID"
	labelNationalID       = "NATIONALID"
	labelIDCardNum        = "IDCARDNUM"
	labelTaxNum           = "TAXNUM"
	labelLicensePlateNum  = "LICENSEPLATENUM"
	labelPassword         = "PASSWORD"
	labelIBAN             = "IBAN"
	labelAge              = "AGE"
	labelSecurityToken    = "SECURITYTOKEN"
	labelCreditCardNumber = "CREDITCARDNUMBER"
	labelUsername         = "USERNAME"
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
		labelSurname:          func(original string) string { return piiGenerators.SurnameGenerator(s.rng, original) },
		labelFirstName:        func(original string) string { return piiGenerators.FirstNameGenerator(s.rng, original) },
		labelBuildingNum:      func(original string) string { return piiGenerators.BuildingNumGenerator(s.rng, original) },
		labelDateOfBirth:      func(original string) string { return piiGenerators.DateOfBirthGenerator(s.rng, original) },
		labelEmail:            func(original string) string { return piiGenerators.EmailGenerator(s.rng, original) },
		labelPhoneNumber:      func(original string) string { return piiGenerators.PhoneGenerator(s.rng, original) },
		labelCity:             func(original string) string { return piiGenerators.CityGenerator(s.rng, original) },
		labelURL:              func(original string) string { return piiGenerators.UrlGenerator(s.rng, original) },
		labelCompanyName:      func(original string) string { return piiGenerators.CompanyNameGenerator(s.rng, original) },
		labelState:            func(original string) string { return piiGenerators.StateGenerator(s.rng, original) },
		labelZip:              func(original string) string { return piiGenerators.ZipCodeGenerator(s.rng, original) },
		labelStreet:           func(original string) string { return piiGenerators.StreetGenerator(s.rng, original) },
		labelCountry:          func(original string) string { return piiGenerators.CountryGenerator(s.rng, original) },
		labelSSN:              func(original string) string { return piiGenerators.SSNGenerator(s.rng, original) },
		labelDriverLicenseNum: func(original string) string { return piiGenerators.DriverLicenseNumGenerator(s.rng, original) },
		labelPassportID:       func(original string) string { return piiGenerators.PassportIdGenerator(s.rng, original) },
		labelNationalID:       func(original string) string { return piiGenerators.NationalIdGenerator(s.rng, original) },
		labelIDCardNum:        func(original string) string { return piiGenerators.IDCardNumGenerator(s.rng, original) },
		labelTaxNum:           func(original string) string { return piiGenerators.TaxNumGenerator(s.rng, original) },
		labelLicensePlateNum:  func(original string) string { return piiGenerators.LicensePlateNumGenerator(s.rng, original) },
		labelPassword:         func(original string) string { return piiGenerators.PasswordGenerator(s.rng, original) },
		labelIBAN:             func(original string) string { return piiGenerators.IbanGenerator(s.rng, original) },
		labelAge:              func(original string) string { return piiGenerators.AgeGenerator(s.rng, original) },
		labelSecurityToken:    func(original string) string { return piiGenerators.SecurityTokenGenerator(s.rng, original) },
		labelCreditCardNumber: func(original string) string { return piiGenerators.CreditCardGenerator(s.rng, original) },
		labelUsername:         func(original string) string { return piiGenerators.UsernameGenerator(s.rng, original) },
	}

	if generator, exists := generators[label]; exists {
		return generator
	}

	// Return generic generator for unknown labels
	return func(original string) string { return piiGenerators.GenericGenerator(s.rng, original) }
}
