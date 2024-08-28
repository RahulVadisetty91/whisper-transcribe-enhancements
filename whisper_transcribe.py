import argparse
import os
import json
import numpy as np
import torch
import tqdm
from typing import List, Optional, Tuple, Union

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, get_tokenizer
from .utils import (
    exact_div,
    format_timestamp,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)

if TYPE_CHECKING:
    from .model import Whisper


def load_model(model_size: str, device: str, fp16: bool) -> "Whisper":
    from .model import Whisper
    return Whisper(model_size=model_size, device=device, fp16=fp16)


def process_audio(audio: Union[str, np.ndarray, torch.Tensor], model: "Whisper") -> torch.Tensor:
    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    return mel


def decode_segment(mel_segment: torch.Tensor, model: "Whisper", decode_options: dict) -> DecodingResult:
    decode_result = model.decode(mel_segment, DecodingOptions(**decode_options))
    return decode_result


def format_output(all_segments: List[dict], output_format: str, output_dir: str) -> None:
    if output_format == "json":
        output_file = os.path.join(output_dir, "transcription.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, ensure_ascii=False, indent=2)
    elif output_format == "txt":
        output_file = os.path.join(output_dir, "transcription.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in all_segments:
                f.write(f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}\n")
                f.write(f"{segment['text']}\n\n")


def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    output_format: str = "json",
    output_dir: str = ".",
    verbose: Optional[bool] = False,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    fp16: bool = True
):
    mel = process_audio(audio, model)
    dtype = torch.float16 if fp16 else torch.float32
    if dtype == torch.float16 and model.device == torch.device("cpu"):
        warnings.warn("FP16 is not supported on CPU; using FP32 instead")
        dtype = torch.float32

    decode_options = {
        "temperature": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "language": "en",  # Default language
        "task": "transcribe",
    }

    tokenizer = get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe"
    )

    clip_timestamps = [float(ts) for ts in clip_timestamps.split(",") if ts]
    seek_points = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) % 2 == 1:
        seek_points.append(mel.shape[-1])
    seek_clips = list(zip(seek_points[::2], seek_points[1::2]))

    all_segments = []

    with tqdm.tqdm(total=mel.shape[-1], unit="frames", disable=verbose is not False) as pbar:
        for seek_clip_start, seek_clip_end in seek_clips:
            mel_segment = mel[:, seek_clip_start:seek_clip_end]
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)
            result = decode_segment(mel_segment, model, decode_options)

            if no_speech_threshold is not None and result.no_speech_prob > no_speech_threshold:
                continue

            segment = {
                "seek": seek_clip_start,
                "start": float(seek_clip_start * HOP_LENGTH / SAMPLE_RATE),
                "end": float(seek_clip_end * HOP_LENGTH / SAMPLE_RATE),
                "text": tokenizer.decode(result.tokens),
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }

            if word_timestamps:
                segment["words"] = add_word_timestamps(
                    segment["text"], result.tokens, tokenizer, mel_segment, model,
                    segment["start"], segment["end"], **decode_options
                )
                if hallucination_silence_threshold is not None:
                    segment["words"] = [
                        word for word in segment["words"]
                        if word["start"] >= segment["start"]
                    ]
                else:
                    segment["words"] = [
                        word for word in segment["words"]
                        if word["end"] >= segment["start"]
                    ]
            all_segments.append(segment)
            pbar.update(mel_segment.shape[-1])

    format_output(all_segments, output_format, output_dir)

    if verbose:
        for segment in all_segments:
            print(f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}")
            print(segment['text'])
            print()

    if hallucination_silence_threshold:
        return [
            segment for segment in all_segments
            if any(word["start"] >= segment["start"] for word in segment.get("words", []))
        ]
    return all_segments


def cli():
    parser = argparse.ArgumentParser(description="Transcribe or translate audio using the Whisper model.")
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to the audio file or directory containing audio files."
    )
    parser.add_argument(
        "--model", type=str, default="base", choices=["base", "small", "medium", "large"],
        help="Specify the model size to use."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"],
        help="Device to use for inference."
    )
    parser.add_argument(
        "--output-format", type=str, default="json", choices=["json", "txt"],
        help="Format of the transcription output."
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save transcription output."
    )
    parser.add_argument(
        "--verbose", type=str2bool, nargs='?', const=True, default=False,
        help="Print detailed information about the transcription process."
    )
    parser.add_argument(
        "--initial-prompt", type=str, default=None,
        help="Initial prompt to guide the transcription or translation."
    )
    parser.add_argument(
        "--temperature", type=float, nargs='*', default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Temperature settings for decoding."
    )
    parser.add_argument(
        "--compression-ratio-threshold", type=float, default=2.4,
        help="Compression ratio threshold for fallback."
    )
    parser.add_argument(
        "--logprob-threshold", type=float, default=-1.0,
        help="Log probability threshold for fallback."
    )
    parser.add_argument(
        "--no-speech-threshold", type=float, default=0.6,
        help="No-speech probability threshold for skipping segments."
    )
    parser.add_argument(
        "--word-timestamps", type=str2bool, nargs='?', const=True, default=False,
        help="Include word-level timestamps in the output."
    )
    parser.add_argument(
        "--clip-timestamps", type=str, default="0",
        help="Comma-separated list of clip timestamps (in seconds) to process."
    )
    parser.add_argument(
        "--hallucination-silence-threshold", type=float, default=None,
        help="Threshold for detecting hallucinations by measuring silence duration."
    )
    parser.add_argument(
        "--fp16", type=str2bool, nargs='?', const=True, default=True,
        help="Use FP16 precision for inference."
    )
    args = parser.parse_args()

    model = load_model(model_size=args.model, device=args.device, fp16=args.fp16)

    if os.path.isdir(args.audio):
        audio_files = [os.path.join(args.audio, f) for f in os.listdir(args.audio) if f.endswith(".wav")]
    else:
        audio_files = [args.audio]

    for audio_file in audio_files:
        audio = np.load(audio_file)
        transcription = transcribe(
            model=model,
            audio=audio,
            verbose=args.verbose,
            temperature=args.temperature,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            initial_prompt=args.initial_prompt,
            word_timestamps=args.word_timestamps,
            clip_timestamps=args.clip_timestamps,
            hallucination_silence_threshold=args.hallucination_silence_threshold,
            output_format=args.output_format,
            output_dir=args.output_dir,
        )
        print(f"Transcription saved to {args.output_dir}")


if __name__ == "__main__":
    cli()
