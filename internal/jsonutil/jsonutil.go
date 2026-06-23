package jsonutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

const maxJSONDepth = 1024

func RejectDuplicateKeys(data []byte, label string) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	if err := rejectDuplicateValue(dec, label, 0); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		if err == nil {
			return fmt.Errorf("%s: trailing JSON data", label)
		}
		return err
	}
	return nil
}

func rejectDuplicateValue(dec *json.Decoder, label string, depth int) error {
	if depth > maxJSONDepth {
		return fmt.Errorf("%s: JSON nesting too deep", label)
	}
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]struct{}{}
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return err
			}
			key, ok := keyTok.(string)
			if !ok {
				return fmt.Errorf("%s: invalid JSON object key", label)
			}
			if _, exists := seen[key]; exists {
				return fmt.Errorf("%s: duplicate JSON key %q", label, key)
			}
			seen[key] = struct{}{}
			if err := rejectDuplicateValue(dec, label, depth+1); err != nil {
				return err
			}
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		if end != json.Delim('}') {
			return fmt.Errorf("%s: invalid JSON object", label)
		}
	case '[':
		for dec.More() {
			if err := rejectDuplicateValue(dec, label, depth+1); err != nil {
				return err
			}
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		if end != json.Delim(']') {
			return fmt.Errorf("%s: invalid JSON array", label)
		}
	default:
		return fmt.Errorf("%s: invalid JSON delimiter %q", label, delim)
	}
	return nil
}
