package pii

import (
	"fmt"
	"math/rand"
)

// Standard dummy data generators for various PII types

// EmailGenerator generates dummy email addresses
func EmailGenerator(rng *rand.Rand, original string) string {
	firstNames := []string{"jane", "john", "alex", "sam", "taylor", "casey", "jordan", "riley"}
	lastNames := []string{"doe", "smith", "johnson", "brown", "davis", "wilson", "moore", "taylor"}
	domains := []string{"example.com", "test.com", "demo.org", "sample.net"}

	firstName := firstNames[rng.Intn(len(firstNames))]
	lastName := lastNames[rng.Intn(len(lastNames))]
	domain := domains[rng.Intn(len(domains))]

	return fmt.Sprintf("%s.%s@%s", firstName, lastName, domain)
}

// PhoneGenerator generates dummy phone numbers
func PhoneGenerator(rng *rand.Rand, original string) string {
	// Generate a random 3-digit area code (200-999)
	areaCode := 200 + rng.Intn(800)

	// Generate a random 3-digit exchange (200-999)
	exchange := 200 + rng.Intn(800)

	// Generate a random 4-digit number
	number := 1000 + rng.Intn(9000)

	// Randomly choose format
	formats := []string{"%d-%d-%d", "%d.%d.%d", "(%d) %d-%d"}
	format := formats[rng.Intn(len(formats))]

	return fmt.Sprintf(format, areaCode, exchange, number)
}

// SSNGenerator generates dummy SSN numbers (SOCIALNUM)
func SSNGenerator(rng *rand.Rand, original string) string {
	// Generate random numbers, avoiding obvious patterns
	first := 100 + rng.Intn(900)   // 100-999
	second := 10 + rng.Intn(90)    // 10-99
	third := 1000 + rng.Intn(9000) // 1000-9999

	return fmt.Sprintf("%d-%d-%d", first, second, third)
}

// CreditCardGenerator generates dummy credit card numbers
func CreditCardGenerator(rng *rand.Rand, original string) string {
	// Generate 4 groups of 4 digits
	groups := make([]int, 4)
	for i := range groups {
		groups[i] = 1000 + rng.Intn(9000)
	}

	// Randomly choose format
	formats := []string{"%d %d %d %d", "%d-%d-%d-%d"}
	format := formats[rng.Intn(len(formats))]

	return fmt.Sprintf(format, groups[0], groups[1], groups[2], groups[3])
}

// UsernameGenerator generates dummy usernames
func UsernameGenerator(rng *rand.Rand, original string) string {
	prefixes := []string{"user", "person", "member", "account", "demo"}
	numbers := 1000 + rng.Intn(9000)

	prefix := prefixes[rng.Intn(len(prefixes))]
	return fmt.Sprintf("%s%d", prefix, numbers)
}

// DateOfBirthGenerator generates dummy dates of birth
func DateOfBirthGenerator(rng *rand.Rand, original string) string {
	// Generate a date between 1950 and 2005
	year := 1950 + rng.Intn(55)
	month := 1 + rng.Intn(12)
	day := 1 + rng.Intn(28) // Keep it simple to avoid invalid dates

	// Try to match the format of the original
	if len(original) > 0 {
		if original[2] == '/' || original[2] == '-' {
			sep := string(original[2])
			return fmt.Sprintf("%02d%s%02d%s%d", month, sep, day, sep, year)
		}
	}

	return fmt.Sprintf("%02d/%02d/%d", month, day, year)
}

// StreetGenerator generates dummy street addresses
func StreetGenerator(rng *rand.Rand, original string) string {
	streetNames := []string{"Main St", "Oak Ave", "Maple Dr", "Park Blvd", "Elm Street", "Pine Road", "Cedar Lane", "Washington St"}
	number := 100 + rng.Intn(9900)

	street := streetNames[rng.Intn(len(streetNames))]
	return fmt.Sprintf("%d %s", number, street)
}

// ZipCodeGenerator generates dummy zip codes
func ZipCodeGenerator(rng *rand.Rand, original string) string {
	// Generate 5-digit zip code
	zipCode := 10000 + rng.Intn(89999)

	// Check if original has ZIP+4 format
	if len(original) > 5 && (original[5] == '-') {
		extension := 1000 + rng.Intn(8999)
		return fmt.Sprintf("%05d-%04d", zipCode, extension)
	}

	return fmt.Sprintf("%05d", zipCode)
}

// CityGenerator generates dummy city names
func CityGenerator(rng *rand.Rand, original string) string {
	cities := []string{"Springfield", "Riverside", "Greenville", "Fairview", "Madison", "Georgetown", "Salem", "Arlington"}
	return cities[rng.Intn(len(cities))]
}

// BuildingNumGenerator generates dummy building numbers
func BuildingNumGenerator(rng *rand.Rand, original string) string {
	number := 1 + rng.Intn(999)

	// Sometimes add a letter suffix
	if rng.Float32() < 0.3 {
		letter := string(rune('A' + rng.Intn(6)))
		return fmt.Sprintf("%d%s", number, letter)
	}

	return fmt.Sprintf("%d", number)
}

// FirstNameGenerator generates dummy first names
func FirstNameGenerator(rng *rand.Rand, original string) string {
	names := []string{"John", "Jane", "Michael", "Sarah", "David", "Emily", "James", "Emma", "Robert", "Olivia"}
	return names[rng.Intn(len(names))]
}

// SurnameGenerator generates dummy last names
func SurnameGenerator(rng *rand.Rand, original string) string {
	surnames := []string{"Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Wilson"}
	return surnames[rng.Intn(len(surnames))]
}

// IDCardNumGenerator generates dummy ID card numbers
func IDCardNumGenerator(rng *rand.Rand, original string) string {
	// Generate a random ID format: XX-XXXXXXX
	prefix := rng.Intn(90) + 10
	number := 1000000 + rng.Intn(8999999)

	return fmt.Sprintf("%02d-%07d", prefix, number)
}

// DriverLicenseNumGenerator generates dummy driver's license numbers
func DriverLicenseNumGenerator(rng *rand.Rand, original string) string {
	// Generate format: A123456789
	letter := string(rune('A' + rng.Intn(26)))
	number := 100000000 + rng.Intn(899999999)

	return fmt.Sprintf("%s%09d", letter, number)
}

// AccountNumGenerator generates dummy account numbers
func AccountNumGenerator(rng *rand.Rand, original string) string {
	// Generate 10-12 digit account number
	length := 10 + rng.Intn(3)
	min := 1
	max := 1
	for i := 0; i < length; i++ {
		min *= 10
		max *= 10
	}
	min /= 10
	max -= 1

	number := min + rng.Intn(max-min)
	return fmt.Sprintf("%d", number)
}

// TaxNumGenerator generates dummy tax identification numbers
func TaxNumGenerator(rng *rand.Rand, original string) string {
	// Generate format: XX-XXXXXXX (EIN-like format)
	first := 10 + rng.Intn(89)
	second := 1000000 + rng.Intn(8999999)

	return fmt.Sprintf("%02d-%07d", first, second)
}

// GenericGenerator is a fallback generator for unknown types
func GenericGenerator(rng *rand.Rand, original string) string {
	return "[REDACTED]"
}
