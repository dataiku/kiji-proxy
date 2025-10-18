package pii

// PIIPatterns defines regex patterns for various PII types
var PIIPatterns = map[string]string{
	"EMAIL":            `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`,
	"TELEPHONENUM":     `\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b`,
	"SOCIALNUM":        `\b\d{3}-\d{2}-\d{4}\b`,
	"CREDITCARDNUMBER": `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`,
	"USERNAME":         `\b(?:username|user|login)[\s:=]+([a-zA-Z0-9_-]{3,20})\b`,
	"DATEOFBIRTH":      `\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b`,
	"ZIPCODE":          `\b\d{5}(?:-\d{4})?\b`,
	"ACCOUNTNUM":       `\b(?:account|acct)[\s#:]*(\d{8,12})\b`,
	"IDCARDNUM":        `\b(?:ID|id)[\s#:]*([A-Z0-9]{6,12})\b`,
	"DRIVERLICENSENUM": `\b(?:DL|license)[\s#:]*([A-Z][0-9]{8,9})\b`,
	"TAXNUM":           `\b\d{2}-\d{7}\b`,
}
