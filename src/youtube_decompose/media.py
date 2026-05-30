from __future__ import annotations

from pathlib import Path


def convert_video_to_images(
    video_path: str | Path,
    work_dir: str | Path,
    frame_rate: int = 10,
) -> int:
    """Sample video frames into work_dir/image_temp and return frame count."""

    image_dir = Path(work_dir) / "image_temp"
    image_dir.mkdir(parents=True, exist_ok=True)

    import pliers as pl
    from tqdm import tqdm
    from pliers.filters import FrameSamplingFilter
    from pliers.stimuli import VideoStim

    pl.set_options(use_generators=True, cache_transformers=False)
    video_stim = VideoStim(str(video_path))
    try:
        frame_filter = FrameSamplingFilter(hertz=frame_rate)
        frame_list = frame_filter.transform(video_stim)

        for image in tqdm(frame_list.frames, total=frame_list.n_frames):
            image.save(str(image_dir / f"image_split-{image.frame_num}.png"))

        return int(frame_list.n_frames)
    finally:
        video_stim.clip.close()


def convert_video_to_audio(video_path: str | Path, work_dir: str | Path) -> Path:
    """Extract full WAV audio into work_dir/audio_temp and return its path."""

    audio_dir = Path(work_dir) / "audio_temp"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio_full.wav"

    import pliers as pl
    from pydub import AudioSegment
    from pliers.converters import VideoToAudioConverter
    from pliers.stimuli import VideoStim

    pl.set_options(use_generators=True, cache_transformers=False)
    video_stim = VideoStim(str(video_path))
    audio_stim = None

    try:
        converter = VideoToAudioConverter()
        audio_stim = converter.transform(video_stim)
        audio_stim.save(str(audio_path))

        audio_segment = AudioSegment.from_file(audio_path)
        if audio_segment.channels == 2:
            left, right = audio_segment.split_to_mono()
            left.export(audio_dir / "audio_full_left.wav", format="wav")
            right.export(audio_dir / "audio_full_right.wav", format="wav")

        return audio_path
    finally:
        video_stim.clip.close()
        if audio_stim is not None:
            audio_stim.clip.close()
