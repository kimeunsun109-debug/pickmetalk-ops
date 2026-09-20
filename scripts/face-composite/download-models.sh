#!/usr/bin/env bash
# Download MediaPipe models for v8 face compositing (run once).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/models" && pwd)"
mkdir -p "$DIR"
curl -sL -o "$DIR/face_landmarker.task" \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
curl -sL -o "$DIR/hair_segmenter.tflite" \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
curl -sL -o "$DIR/selfie_multiclass.tflite" \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
echo "Models saved to $DIR"
