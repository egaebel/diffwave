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

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from diffwave.learner import train
from diffwave.learner import train_distributed
from diffwave.params import load_preset
from diffwave.params import params


def _get_free_port():
    import socketserver

    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def main(args):
    params_to_use = params
    if args.preset is not None:
        params_to_use = load_preset(args.preset)
    replica_count = device_count()
    if replica_count > 1 and args.distributed:
        if params_to_use.batch_size % replica_count != 0:
            raise ValueError(
                f"Batch size {params_to_use.batch_size} is not evenly divisble by # GPUs {replica_count}."
            )
        params_to_use.batch_size = params_to_use.batch_size // replica_count
        port = _get_free_port()
        spawn(
            train_distributed,
            args=(replica_count, port, args, params_to_use),
            nprocs=replica_count,
            join=True,
        )
    else:
        train(args, params_to_use)


if __name__ == "__main__":
    parser = ArgumentParser(description="train (or resume training) a DiffWave model")
    parser.add_argument(
        "model_dir",
        help="directory in which to store model checkpoints and training logs",
    )
    parser.add_argument(
        "data_dirs",
        nargs="+",
        help="space separated list of directories from which to read .wav files for training",
    )
    parser.add_argument(
        "--spec_filename_suffix",
        default=None,
        type=str,
        help="the suffix at the end of file names that hold the mel spectrograms.",
    )
    parser.add_argument(
        "--duplicates_suffix_regex",
        default=None,
        type=str,
        help="the suffix at the end of duplicate mel spectrogram file names. If passed then there "
        "is expected to be multiple spectrograms for each 'wav' file and this suffix is used to convert "
        "from spectrograms with a suffix at the end to the base wave file.",
    )
    parser.add_argument(
        "--max_steps", default=None, type=int, help="maximum number of training steps"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="use 16-bit floating point operations for training",
    )
    parser.add_argument(
        "--preset",
        default=None,
        type=str,
        help="file path to a json preset file to use for training/data parameters.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="file path to a checkpoint to load an existing model from.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="use multiple GPUs (if available) for training",
    )
    main(parser.parse_args())
