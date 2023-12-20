import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "/mnt/nas/users/lsj/music/llark"
os.environ["GCS_BUCKET_NAME"] = "/mnt/nas/users/lsj/music/llark"

import jsonlines
from m2t.audio_io import convert_to_wav
from m2t.dataset_utils import DATASET_INFO
import soundfile as sf
import io
import math
import numpy as np

def read_element_wav(
    elem,
    audio_dir,
    dataset_info,
    target_sr=44100,
    duration=None):

    track_id = elem[dataset_info.id_col]
    filepath = dataset_info.id_to_filename(track_id, audio_dir)
    samples, sr = read_wav(filepath=filepath, target_sr=target_sr, duration=duration)
    elem["audio"] = samples
    elem["audio_sample_rate"] = sr
    return elem


def drop_audio_features(elem):
    if "audio" in elem:
        del elem["audio"]
    if "audio_sample_rate" in elem:
        del elem["audio_sample_rate"]
    return elem



def read_wav(
    filepath: str, target_sr: int = 44100, duration = None
):
    """Read a wav file, either local on on GCS."""

    print(f"reading audio from {filepath}")
   
    # Case: local file
    with open(filepath, "rb") as f:
        bytes_as_string = f.read()

    # Samplerate does not allow to specify sr when reading; if desired,
    # the audio will need to be resampled in a postprocessing step.
    # For some reason, librosa fails to read due to an issue with
    # lazy-loading of modules when executed within a beam pipeline.
    samples, audio_sr = sf.read(
        io.BytesIO(bytes_as_string),
        frames=math.floor(target_sr * duration) if duration is not None else -1,
    )
    print(
        f"finished reading audio from {filepath} with sr {audio_sr} "
        f"with duration {round(len(samples)/audio_sr,2)}secs"
    )

    if audio_sr != target_sr:
        print(f"resampling audio input {filepath} from {audio_sr} to {target_sr}")
        samples = librosa.resample(samples, orig_sr=audio_sr, target_sr=target_sr)

    assert np.issubdtype(
        samples.dtype, float
    ), f"exected floating-point audio; got type {samples.dtype}"

    return samples, target_sr



raw_dir = "data/fma_small/000" # raw audio path (.mp3)
audio_dir = "data/tmp" # path to directory containing wav audio
annotation_jsonl = "data/annotation.jsonl" # path to annotation wav file
elems = [] # e.g. {"track.id":"000002"}

dataset_name = "fma"
dataset_info = DATASET_INFO[dataset_name]
max_audio_duration_seconds = 360
print(dataset_info)

######################### Starting Data Preprocessing ########################

# ================== 1. convert audio (mp3) to wav ==================
# for raw_audio_name in os.listdir(raw_dir):
#     track_id = raw_audio_name.split('.')[0]
#     elems.append({"track.id": track_id})
#     raw_audio_path = os.path.join(raw_dir, raw_audio_name)
#     convert_to_wav(raw_audio_path, audio_dir) 

# ================== 2. read elements from wav file ==================


# for i in range(len(elems)):
#     elems[i] = read_element_wav(
#             elem=elems[i],
#             audio_dir=audio_dir,
#             duration=max_audio_duration_seconds,
#             dataset_info=dataset_info,)

# print(elems)


# ================== 3. annotation ==================

# from m2t.annotation import (
#     ExtractMadmomChordEstimates,
#     ExtractMadmomDownbeatFeatures,
#     ExtractMadmomKeyEstimates,
#     ExtractMadmomTempoFeatures,
# )


# Extract_List = [ExtractMadmomChordEstimates, ExtractMadmomDownbeatFeatures, ExtractMadmomKeyEstimates, ExtractMadmomTempoFeatures]

# for i in range(len(elems)):
#     for step in Extract_List:
#         elems[i] = step().process(elems[i])[0]

#     elems[i] = drop_audio_features(elems[i])


# print(elems)

# with jsonlines.open(annotation_jsonl, "w") as f:
#     for elem in elems:
#         f.write(elem)


# ================== 4. jsonify dataset (get metadata from csv into jsonl file) ==================
# os.system("python scripts/preprocessing/jsonify_dataset.py \
#             --dataset fma \
#             --input-dir data/fma_metadata \
#             --output-dir data/tmp \
#             --split train")

# ================== 5. merge jsonl files from annotation and metadata   ==================

# os.system("python scripts/preprocessing/merge_jsonl.py \
#                 --annotation-json-file data/annotation.jsonl \
#                 --metadata-json-file data/tmp/fma-train.json \
#                 --output-file data/tmp/final.json")

# ================== 6. obtain Q&A response from chatgpt   ==================
# obtain a qa_json_file
# multiple {"question": xx, "answer": xx} for a track id
# ================== 7. obtain audio representations from jukebox   ==================
# os.system("python jukebox/main.py \
#     --input_dir data/fma_small/000 \
#     --output_dir data/fma_rep")



# ================== 8. merge audio encoding and Q&A response to form final dataset   ==================
# {
# "__key__": [1, 2, ..], "audio_encoding": [arr1, arr2, ...], "audio_encoding_shape": [shape1, shap2, ...], 
# "json": [{"response": [{"question": xx, "answer": xx},
#                       {"question": xx, "answer": xx},
#                      ]
#          },
#          {"response": [{"question": xx, "answer": xx},
#                       {"question": xx, "answer": xx},
#                      ]
#          }
#           ...
#            ]
#}







