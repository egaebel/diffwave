#!/bin/sh

python -m diffwave checkpoints-$(date +%s) tacotron2-ljspeech-cranked-up-preprocessed-mels-dataset --spec_filename_suffix=".npy" --fp16 --preset=presets/ljspeech-cranked-up.json
