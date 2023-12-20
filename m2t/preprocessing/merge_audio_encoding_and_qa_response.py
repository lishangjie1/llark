

import os
import sys
import jsonlines
import numpy as np
QA_RESPONSE_DIR = sys.argv[1]
AUDIO_ENCODING_DIR = sys.argv[2]
OUTPUT = sys.argv[3]

dataset_keys = ["__key__", "audio_encoding", "audio_encoding_shape", "json"]

with jsonlines.open(OUTPUT, "w") as fw:
    
    for fname in os.listdir(QA_RESPONSE_DIR):
        sample = {}
        track_id = fname.split('.')[0]
        audio_encoding_file = f"{track_id}.npy"

        audio_encoding = np.load(AUDIO_ENCODING_DIR + "/" + audio_encoding_file)
        audio_encoding_shape = audio_encoding.shape

        # flattened list of floats
        audio_encoding = audio_encoding.flatten().tolist()

        response_file = f"{track_id}.jsonl"
        elem_json = {"response": []}
        with jsonlines.open(QA_RESPONSE_DIR + "/" + response_file) as f:
            for response in f:
                question, answer = response["question"], response["answer"]
                elem_json["response"].append({"question": question, "answer": answer})
        

        sample["__key__"] = int(track_id)
        sample["audio_encoding"] = audio_encoding
        sample["audio_encoding_shape"] = audio_encoding_shape
        sample["json"] = elem_json


        fw.write(sample)


# print(dataset)







