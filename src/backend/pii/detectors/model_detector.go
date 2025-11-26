package pii

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
)

// RegexDetector implements DetectorClass using regular expressions
type ModelDetector struct {
	baseURL string
}

func NewModelDetector(baseURL string) *ModelDetector {
	return &ModelDetector{
		baseURL: baseURL,
	}
}

// GetName returns the name of this detector
func (m *ModelDetector) GetName() string {
	return "model_detector"
}

// Detect processes the input and returns detected entities
func (m *ModelDetector) Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error) {
	var entities []Entity

	// send input to model server using POST request -> baseURL / detect
	requestBody := map[string]interface{}{
		"text": input.Text,
	}
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return DetectorOutput{}, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/detect", bytes.NewBuffer(jsonData))
	if err != nil {
		return DetectorOutput{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	response, err := client.Do(req)
	if err != nil {
		return DetectorOutput{}, err
	}
	defer func() { _ = response.Body.Close() }()

	// convert the response body to Entities
	entities, err = convertResponseToEntities(response)
	if err != nil {
		return DetectorOutput{}, err
	}

	return DetectorOutput{
		Text:     input.Text,
		Entities: entities,
	}, nil
}

func convertResponseToEntities(response *http.Response) ([]Entity, error) {
	var responseBody map[string]interface{}
	err := json.NewDecoder(response.Body).Decode(&responseBody)
	if err != nil {
		return []Entity{}, err
	}

	entitiesJSON, err := json.Marshal(responseBody["entities"])
	if err != nil {
		return []Entity{}, nil
	}
	var entitiesArray []map[string]interface{}
	err = json.Unmarshal(entitiesJSON, &entitiesArray)
	if err != nil {
		return []Entity{}, err
	}
	var entities []Entity
	for _, entity := range entitiesArray {
		// Handle type conversion for start_pos and end_pos which might be float64 from JSON
		var startPos, endPos int
		if sp, ok := entity["start_pos"].(float64); ok {
			startPos = int(sp)
		} else if sp, ok := entity["start_pos"].(int); ok {
			startPos = sp
		}

		if ep, ok := entity["end_pos"].(float64); ok {
			endPos = int(ep)
		} else if ep, ok := entity["end_pos"].(int); ok {
			endPos = ep
		}

		entities = append(entities, Entity{
			Text:       entity["text"].(string),
			Label:      entity["label"].(string),
			StartPos:   startPos,
			EndPos:     endPos,
			Confidence: entity["confidence"].(float64),
		})
	}
	return entities, nil
}

// Close implements the Detector interface
func (m *ModelDetector) Close() error {
	// Model detector doesn't need cleanup
	return nil
}
