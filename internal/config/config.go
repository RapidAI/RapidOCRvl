package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"paddleocrvl-go/internal/jsonutil"
)

const maxConfigBytes = 16 << 20

type Vision struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	ImageSize         int     `json:"image_size"`
	PatchSize         int     `json:"patch_size"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
	SpatialMergeSize  int     `json:"spatial_merge_size"`
}

type RopeScaling struct {
	MropeSection []int  `json:"mrope_section"`
	Type         string `json:"type"`
	RopeType     string `json:"rope_type"`
}

type Config struct {
	VocabSize          int          `json:"vocab_size"`
	HiddenSize         int          `json:"hidden_size"`
	IntermediateSize   int          `json:"intermediate_size"`
	MaxPositionEmb     int          `json:"max_position_embeddings"`
	NumHiddenLayers    int          `json:"num_hidden_layers"`
	NumAttentionHeads  int          `json:"num_attention_heads"`
	NumKeyValueHeads   int          `json:"num_key_value_heads"`
	HeadDim            int          `json:"head_dim"`
	RMSNormEps         float64      `json:"rms_norm_eps"`
	RopeTheta          float64      `json:"rope_theta"`
	PadTokenID         int          `json:"pad_token_id"`
	ImageTokenID       int          `json:"image_token_id"`
	VisionStartTokenID int          `json:"vision_start_token_id"`
	VisionEndTokenID   int          `json:"vision_end_token_id"`
	VideoTokenID       int          `json:"video_token_id"`
	VisionConfig       Vision       `json:"vision_config"`
	RopeScaling        *RopeScaling `json:"rope_scaling"`
}

func Load(dir string) (*Config, error) {
	path := filepath.Join(dir, "config.json")
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if st.Size() > maxConfigBytes {
		return nil, fmt.Errorf("config.json too large: %d bytes > %d", st.Size(), maxConfigBytes)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if err := jsonutil.RejectDuplicateKeys(b, path); err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	if c.HeadDim == 0 && c.NumAttentionHeads != 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.NumKeyValueHeads == 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}
	if c.RMSNormEps == 0 {
		c.RMSNormEps = 1e-6
	}
	return &c, nil
}
