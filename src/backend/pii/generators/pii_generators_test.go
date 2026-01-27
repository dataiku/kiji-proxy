package pii

import (
	"fmt"
	"math/rand"
	"regexp"
	"strings"
	"testing"
)

// Helper function to create a seeded random generator
func getTestRand(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}

func TestEmailGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := EmailGenerator(rng, "")

	// Check that result matches email format
	emailPattern := regexp.MustCompile(`^[a-z]+\.[a-z]+@(example|test|invalid)\.(com|org|net)$`)
	if !emailPattern.MatchString(result) {
		t.Errorf("EmailGenerator returned invalid email format: %s", result)
	}

	// Check that it contains an @
	if !strings.Contains(result, "@") {
		t.Errorf("EmailGenerator returned email without @: %s", result)
	}

	// Check that domain is reserved/test domain
	validDomains := []string{"example.com", "example.org", "example.net", "test.com", "test.org", "test.net", "invalid.com", "invalid.org"}
	hasValidDomain := false
	for _, domain := range validDomains {
		if strings.HasSuffix(result, domain) {
			hasValidDomain = true
			break
		}
	}
	if !hasValidDomain {
		t.Errorf("EmailGenerator returned email with non-test domain: %s", result)
	}
}

func TestPhoneGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := PhoneGenerator(rng, "")

	// Check that result contains digits and separators
	cleaned := strings.ReplaceAll(result, " ", "")
	cleaned = strings.ReplaceAll(cleaned, "(", "")
	cleaned = strings.ReplaceAll(cleaned, ")", "")
	cleaned = strings.ReplaceAll(cleaned, "-", "")
	cleaned = strings.ReplaceAll(cleaned, ".", "")

	if len(cleaned) != 10 {
		t.Errorf("PhoneGenerator returned phone number with incorrect length: %s (cleaned: %s)", result, cleaned)
	}
}

func TestSSNGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := SSNGenerator(rng, "")

	// Check SSN format: XXX-XX-XXXX
	ssnPattern := regexp.MustCompile(`^\d{3}-\d{2}-\d{4}$`)
	if !ssnPattern.MatchString(result) {
		t.Errorf("SSNGenerator returned invalid SSN format: %s", result)
	}
}

func TestCreditCardGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := CreditCardGenerator(rng, "")

	// Remove separators and check length
	cleaned := strings.ReplaceAll(result, " ", "")
	cleaned = strings.ReplaceAll(cleaned, "-", "")

	if len(cleaned) != 16 {
		t.Errorf("CreditCardGenerator returned credit card with incorrect length: %s (cleaned: %s)", result, cleaned)
	}

	// Check that all are digits
	for _, char := range cleaned {
		if char < '0' || char > '9' {
			t.Errorf("CreditCardGenerator returned non-digit character: %s", result)
			break
		}
	}
}

func TestUsernameGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := UsernameGenerator(rng, "")

	// Check that result contains a prefix and numbers
	usernamePattern := regexp.MustCompile(`^[a-z]+\d{4}$`)
	if !usernamePattern.MatchString(result) {
		t.Errorf("UsernameGenerator returned invalid username format: %s", result)
	}
}

func TestDateOfBirthGenerator(t *testing.T) {
	rng := getTestRand(42)
	
	// Test with empty original
	result := DateOfBirthGenerator(rng, "")
	datePattern := regexp.MustCompile(`^\d{2}/\d{2}/\d{4}$`)
	if !datePattern.MatchString(result) {
		t.Errorf("DateOfBirthGenerator returned invalid date format: %s", result)
	}

	// Test with original using different separator
	rng2 := getTestRand(42)
	result2 := DateOfBirthGenerator(rng2, "01-15-1990")
	if !strings.Contains(result2, "-") {
		t.Errorf("DateOfBirthGenerator did not respect original separator: %s", result2)
	}

	// Test with original using slash separator
	rng3 := getTestRand(42)
	result3 := DateOfBirthGenerator(rng3, "01/15/1990")
	if !strings.Contains(result3, "/") {
		t.Errorf("DateOfBirthGenerator did not respect original separator: %s", result3)
	}
}

func TestStreetGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := StreetGenerator(rng, "")

	// Check that result contains a number and street name
	streetPattern := regexp.MustCompile(`^\d+ .+$`)
	if !streetPattern.MatchString(result) {
		t.Errorf("StreetGenerator returned invalid street format: %s", result)
	}
}

func TestZipCodeGenerator(t *testing.T) {
	rng := getTestRand(42)
	
	// Test basic 5-digit zip
	result := ZipCodeGenerator(rng, "")
	zipPattern := regexp.MustCompile(`^\d{5}$`)
	if !zipPattern.MatchString(result) {
		t.Errorf("ZipCodeGenerator returned invalid zip format: %s", result)
	}

	// Test ZIP+4 format when original has it
	rng2 := getTestRand(42)
	result2 := ZipCodeGenerator(rng2, "12345-6789")
	zipPlusFourPattern := regexp.MustCompile(`^\d{5}-\d{4}$`)
	if !zipPlusFourPattern.MatchString(result2) {
		t.Errorf("ZipCodeGenerator did not generate ZIP+4 format: %s", result2)
	}
}

func TestCityGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := CityGenerator(rng, "")

	// Check that result is non-empty and starts with a capital letter
	if len(result) == 0 {
		t.Errorf("CityGenerator returned empty string")
	}

	if result[0] < 'A' || result[0] > 'Z' {
		t.Errorf("CityGenerator returned city name not starting with capital: %s", result)
	}
}

func TestBuildingNumGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := BuildingNumGenerator(rng, "")

	// Check that result is a number or number with letter suffix
	buildingPattern := regexp.MustCompile(`^\d+[A-F]?$`)
	if !buildingPattern.MatchString(result) {
		t.Errorf("BuildingNumGenerator returned invalid building number format: %s", result)
	}
}

func TestFirstNameGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := FirstNameGenerator(rng, "")

	// Check that result is non-empty and starts with a capital letter
	if len(result) == 0 {
		t.Errorf("FirstNameGenerator returned empty string")
	}

	if result[0] < 'A' || result[0] > 'Z' {
		t.Errorf("FirstNameGenerator returned name not starting with capital: %s", result)
	}
}

func TestSurnameGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := SurnameGenerator(rng, "")

	// Check that result is non-empty and starts with a capital letter or apostrophe
	if len(result) == 0 {
		t.Errorf("SurnameGenerator returned empty string")
	}

	firstChar := result[0]
	if firstChar != '\'' && (firstChar < 'A' || firstChar > 'Z') {
		t.Errorf("SurnameGenerator returned surname with unexpected format: %s", result)
	}
}

func TestIDCardNumGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := IDCardNumGenerator(rng, "")

	// Check format: XX-XXXXXXX
	idPattern := regexp.MustCompile(`^\d{2}-\d{7}$`)
	if !idPattern.MatchString(result) {
		t.Errorf("IDCardNumGenerator returned invalid ID format: %s", result)
	}
}

func TestDriverLicenseNumGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := DriverLicenseNumGenerator(rng, "")

	// Check format: A123456789
	dlPattern := regexp.MustCompile(`^[A-Z]\d{9}$`)
	if !dlPattern.MatchString(result) {
		t.Errorf("DriverLicenseNumGenerator returned invalid driver license format: %s", result)
	}
}

func TestTaxNumGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := TaxNumGenerator(rng, "")

	// Check format: XX-XXXXXXX
	taxPattern := regexp.MustCompile(`^\d{2}-\d{7}$`)
	if !taxPattern.MatchString(result) {
		t.Errorf("TaxNumGenerator returned invalid tax number format: %s", result)
	}
}

func TestUrlGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := UrlGenerator(rng, "")

	// Check that result is a valid URL
	if !strings.HasPrefix(result, "https://www.") {
		t.Errorf("UrlGenerator returned URL without https://www. prefix: %s", result)
	}

	// Check that it uses reserved domains
	hasReservedDomain := strings.Contains(result, "example.") || 
		strings.Contains(result, "test.") || 
		strings.Contains(result, "invalid.") || 
		strings.Contains(result, "localhost.")
	if !hasReservedDomain {
		t.Errorf("UrlGenerator returned URL with non-reserved domain: %s", result)
	}
}

func TestCompanyNameGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := CompanyNameGenerator(rng, "")

	// Check that result contains a space (prefix and suffix)
	if !strings.Contains(result, " ") {
		t.Errorf("CompanyNameGenerator returned company name without space: %s", result)
	}

	// Check that result is non-empty
	if len(result) == 0 {
		t.Errorf("CompanyNameGenerator returned empty string")
	}
}

func TestStateGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := StateGenerator(rng, "")

	// Check that result is non-empty
	if len(result) == 0 {
		t.Errorf("StateGenerator returned empty string")
	}
}

func TestCountryGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := CountryGenerator(rng, "")

	// Check that result is non-empty
	if len(result) == 0 {
		t.Errorf("CountryGenerator returned empty string")
	}
}

func TestPassportIdGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := PassportIdGenerator(rng, "")

	// Check format: AB1234567
	passportPattern := regexp.MustCompile(`^[A-Z]{2}\d{7}$`)
	if !passportPattern.MatchString(result) {
		t.Errorf("PassportIdGenerator returned invalid passport format: %s", result)
	}
}

func TestNationalIdGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := NationalIdGenerator(rng, "")

	// Check format: XXX-XXXX-XXXX
	nationalIdPattern := regexp.MustCompile(`^\d{3}-\d{4}-\d{4}$`)
	if !nationalIdPattern.MatchString(result) {
		t.Errorf("NationalIdGenerator returned invalid national ID format: %s", result)
	}
}

func TestLicensePlateNumGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := LicensePlateNumGenerator(rng, "")

	// Check format: ABC-1234
	platePattern := regexp.MustCompile(`^[A-Z]{3}-\d{4}$`)
	if !platePattern.MatchString(result) {
		t.Errorf("LicensePlateNumGenerator returned invalid license plate format: %s", result)
	}
}

func TestPasswordGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := PasswordGenerator(rng, "")

	// Check length is between 12 and 16
	if len(result) < 12 || len(result) > 16 {
		t.Errorf("PasswordGenerator returned password with invalid length: %d", len(result))
	}

	// Check that it contains mixed characters
	hasLower := false
	hasUpper := false
	hasDigit := false

	for _, char := range result {
		if char >= 'a' && char <= 'z' {
			hasLower = true
		} else if char >= 'A' && char <= 'Z' {
			hasUpper = true
		} else if char >= '0' && char <= '9' {
			hasDigit = true
		}
	}

	// Note: Due to randomness, not all passwords will have all types,
	// but the charset allows for all types
	if !hasLower && !hasUpper && !hasDigit {
		t.Errorf("PasswordGenerator returned password without alphanumeric chars: %s", result)
	}
}

func TestIbanGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := IbanGenerator(rng, "")

	// Check format: CC00 0000 0000 0000 0000
	ibanPattern := regexp.MustCompile(`^[A-Z]{2}\d{2} \d{4} \d{4} \d{4} \d{4}$`)
	if !ibanPattern.MatchString(result) {
		t.Errorf("IbanGenerator returned invalid IBAN format: %s", result)
	}
}

func TestAgeGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := AgeGenerator(rng, "")

	// Check that result is a number
	agePattern := regexp.MustCompile(`^\d+$`)
	if !agePattern.MatchString(result) {
		t.Errorf("AgeGenerator returned invalid age format: %s", result)
	}

	// Parse and check range
	var age int
	_, err := fmt.Sscanf(result, "%d", &age)
	if err != nil {
		t.Errorf("AgeGenerator returned non-numeric value: %s", result)
	}

	if age < 18 || age >= 86 {
		t.Errorf("AgeGenerator returned age outside expected range (18-85): %d", age)
	}
}

func TestSecurityTokenGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := SecurityTokenGenerator(rng, "")

	// Check that result has a prefix
	hasPrefixUnderscore := strings.Contains(result, "_")
	if !hasPrefixUnderscore && !strings.HasPrefix(result, "AKIA") && !strings.HasPrefix(result, "ABIA") && !strings.HasPrefix(result, "ACCA") {
		t.Errorf("SecurityTokenGenerator returned token without expected format: %s", result)
	}

	// Check minimum length
	if len(result) < 10 {
		t.Errorf("SecurityTokenGenerator returned token too short: %s", result)
	}
}

func TestGenericGenerator(t *testing.T) {
	rng := getTestRand(42)
	result := GenericGenerator(rng, "some original value")

	expected := "[REDACTED]"
	if result != expected {
		t.Errorf("GenericGenerator returned %s, expected %s", result, expected)
	}
}

// Test that generators produce different outputs with different seeds
func TestGeneratorsDifferentSeeds(t *testing.T) {
	rng1 := getTestRand(1)
	rng2 := getTestRand(2)

	email1 := EmailGenerator(rng1, "")
	email2 := EmailGenerator(rng2, "")

	if email1 == email2 {
		t.Errorf("EmailGenerator produced same output with different seeds: %s", email1)
	}
}

// Test that generators produce consistent outputs with same seed
func TestGeneratorsSameSeed(t *testing.T) {
	rng1 := getTestRand(42)
	rng2 := getTestRand(42)

	email1 := EmailGenerator(rng1, "")
	email2 := EmailGenerator(rng2, "")

	if email1 != email2 {
		t.Errorf("EmailGenerator produced different outputs with same seed: %s vs %s", email1, email2)
	}
}

// Test all generators return non-empty strings
func TestAllGeneratorsNonEmpty(t *testing.T) {
	generators := map[string]func(*rand.Rand, string) string{
		"EmailGenerator":              EmailGenerator,
		"PhoneGenerator":              PhoneGenerator,
		"SSNGenerator":                SSNGenerator,
		"CreditCardGenerator":         CreditCardGenerator,
		"UsernameGenerator":           UsernameGenerator,
		"DateOfBirthGenerator":        DateOfBirthGenerator,
		"StreetGenerator":             StreetGenerator,
		"ZipCodeGenerator":            ZipCodeGenerator,
		"CityGenerator":               CityGenerator,
		"BuildingNumGenerator":        BuildingNumGenerator,
		"FirstNameGenerator":          FirstNameGenerator,
		"SurnameGenerator":            SurnameGenerator,
		"IDCardNumGenerator":          IDCardNumGenerator,
		"DriverLicenseNumGenerator":   DriverLicenseNumGenerator,
		"TaxNumGenerator":             TaxNumGenerator,
		"UrlGenerator":                UrlGenerator,
		"CompanyNameGenerator":        CompanyNameGenerator,
		"StateGenerator":              StateGenerator,
		"CountryGenerator":            CountryGenerator,
		"PassportIdGenerator":         PassportIdGenerator,
		"NationalIdGenerator":         NationalIdGenerator,
		"LicensePlateNumGenerator":    LicensePlateNumGenerator,
		"PasswordGenerator":           PasswordGenerator,
		"IbanGenerator":               IbanGenerator,
		"AgeGenerator":                AgeGenerator,
		"SecurityTokenGenerator":      SecurityTokenGenerator,
		"GenericGenerator":            GenericGenerator,
	}

	for name, generator := range generators {
		result := generator(rand.New(rand.NewSource(42)), "")
		if len(result) == 0 {
			t.Errorf("%s returned empty string", name)
		}
	}
}
