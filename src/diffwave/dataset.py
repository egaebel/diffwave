# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import librosa
import numpy as np
import os
import random
import re
import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        sample_rate,
        spec_filename_suffix=".wav.spec.npy",
        use_torchaudio=False,
        filter_files_by_spectrogram_length=True,
        crop_mel_frames=None,
        duplicates_suffix_regex=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.spec_filename_suffix = spec_filename_suffix
        temp_file_names = []
        for path in paths:
            temp_file_names += glob(
                f"{path}/**/*{spec_filename_suffix}", recursive=True
            )
        self.resampler_fn = None
        self.use_torchaudio = use_torchaudio
        if self.use_torchaudio:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
        self.crop_mel_frames = crop_mel_frames
        self.duplicates_suffix_regex = duplicates_suffix_regex
        print(
            f"Will use regex: '{duplicates_suffix_regex}' for obtaining 'wav' files from duplicate mel spetrogram files."
        )
        if filter_files_by_spectrogram_length and self.crop_mel_frames is not None:
            print(
                f"Filtering '{len(temp_file_names)}' files by spectrogram length with max mel frames: '{self.crop_mel_frames}'.....",
                flush=True,
            )
            filtered_file_names = list(
                filter(
                    lambda x: np.load(x).shape[1] >= self.crop_mel_frames,
                    temp_file_names,
                ),
            )
            print(
                f"Finished filtering files by spectrogram length! Finished with '{len(filtered_file_names)}' files!",
                flush=True,
            )
            self.filenames = filtered_file_names
        else:
            self.filenames = temp_file_names

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        audio_filename = self.filenames[idx]
        audio_filename_without_extension, _ = os.path.splitext(audio_filename)
        spec_filename = f"{audio_filename_without_extension}{self.spec_filename_suffix}"
        """
        spec_filename = self.filenames[idx]
        audio_filename = spec_filename[: -len(self.spec_filename_suffix)]
        if self.duplicates_suffix_regex is not None:
            audio_filename = re.sub(
                self.duplicates_suffix_regex + "$", "", audio_filename
            )
        audio_filename += ".wav"

        spectrogram = np.load(spec_filename)
        if self.use_torchaudio:
            signal, signal_sample_rate = torchaudio.load_wav(audio_filename)
            if signal_sample_rate != self.sample_rate:
                if not self.resampler_fn:
                    self.resampler_fn = torchaudio.transforms.Resample(
                        signal_sample_rate, self.sample_rate
                    )
                signal = self.resampler_fn(signal)
            # Torchaudio doesn't rescale audio to [-1.0, 1.0], so we manually
            # do it here, unlike librosa which automatically rescales in load.
            return {
                "audio": signal[0] / 32767.5,
                "spectrogram": spectrogram.T,
                "audio_file_name": audio_filename,
                "spectrogram_file_name": spec_filename,
            }
        else:
            signal, signal_sample_rate = librosa.core.load(
                audio_filename, sr=self.sample_rate
            )
            signal = torch.unsqueeze(torch.tensor(signal), 0)
            # Librosa will rescale audio automatically to be in [-1.0, 1.0]
            # whereas torchaudio doesn't.
            return {
                "audio": signal[0],
                "spectrogram": spectrogram.T,
                "audio_file_name": audio_filename,
                "spectrogram_file_name": spec_filename,
            }


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        original_minibatch = minibatch
        # print("\n\n\n\nCOLLATE!")
        samples_per_frame = self.params.hop_samples
        for record in minibatch:
            # Filter out records that aren't long enough.
            # NOTE: Do I need this? YES, YES I DO NEED THIS
            if len(record["spectrogram"]) < self.params.crop_mel_frames:
                del record["spectrogram"]
                del record["audio"]
                print("FILTERED OUT TOO SHORT AUDIO.")
                continue

            start = random.randint(
                0,
                record["spectrogram"].shape[0] - self.params.crop_mel_frames,
            )
            end = start + self.params.crop_mel_frames
            record["spectrogram"] = record["spectrogram"][start:end].T

            start *= samples_per_frame
            end *= samples_per_frame
            # print(f"trimming audio record to bounds: start: '{start}' and end: '{end}'")
            record["audio"] = record["audio"][start:end]
            record["audio"] = np.pad(
                record["audio"],
                (0, (end - start) - len(record["audio"])),
                mode="constant",
            )

        audio = None
        spectrogram = None
        spectrogram_records = None
        try:
            # print(f"minibatch: '{minibatch}'", flush=True)
            audio_records = [
                record["audio"] for record in minibatch if "audio" in record
            ]
            # print(f"audio_records: '{audio_records}'", flush=True)
            audio = np.stack(audio_records)
            spectrogram_records = [
                record["spectrogram"] for record in minibatch if "spectrogram" in record
            ]
            # int(f"spectrogram_records: '{spectrogram_records}'", flush=True)
            spectrogram = np.stack(spectrogram_records)
        except ValueError as e:
            print(f"audio: '{audio}'", flush=True)
            print(f"spectrogram: '{spectrogram}'", flush=True)
            print(f"minibatch: '{minibatch}'", flush=True)
            raise ValueError(
                f"ERROR CAUGHT: audio_records: '{audio_records}'\n\n"
                f"spectrogram_records: '{spectrogram_records}'\n\n"
                f"minibatch: '{minibatch}'\n\n"
                f"original_minibatch: '{original_minibatch}'\n\n"
            ) from e
        return {
            "audio": torch.from_numpy(audio),
            "spectrogram": torch.from_numpy(spectrogram),
        }


def from_path(
    data_dirs,
    params,
    is_distributed=False,
    spec_filename_suffix=None,
    duplicates_suffix_regex=None,
):

    num_files = sum(map(lambda x: len(os.listdir(x)), data_dirs))
    print(f"Loading dataset from path: '{data_dirs}' found: '{num_files}' files!")
    if spec_filename_suffix is None:
        dataset = NumpyDataset(
            data_dirs,
            sample_rate=params.sample_rate,
            crop_mel_frames=params.crop_mel_frames,
            duplicates_suffix_regex=duplicates_suffix_regex,
        )
    else:
        print(
            f"Creating dataset using spectrogram file name suffix: '{spec_filename_suffix}'"
        )
        dataset = NumpyDataset(
            data_dirs,
            sample_rate=params.sample_rate,
            spec_filename_suffix=spec_filename_suffix,
            duplicates_suffix_regex=duplicates_suffix_regex,
            crop_mel_frames=params.crop_mel_frames,
        )
    print(f"\nCreating dataloader with dataset length: '{len(dataset)}'.....\n")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
    )
