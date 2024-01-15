
import argparse
import json
import jsonlines
from m2t.dataset_utils import DATASET_INFO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-json-file", required=True, help="input file for the annotation json file from dataset.")
    parser.add_argument("--metadata-json-file", required=True, help="input file for the annotation json file from dataset.")
    parser.add_argument(
        "--output-file",
        default="./tmp/final.jsonl",
        help="where to write the merged json file(s) for the dataset.",
    )

    args = parser.parse_args()

    annotation_file = args.annotation_json_file
    metadata_json_file = args.metadata_json_file
    output_file = args.output_file

    with open(annotation_file, "r") as af, open(metadata_json_file, "r") as mf, jsonlines.open(output_file, "w") as out:
        
        af_content, mf_content, final_content = [], [], []
        af_ids, mf_ids, ids = {}, {}, {}
        for line_idx, line in enumerate(af):
            af_content.append(json.loads(line))
            track_id = int(af_content[-1]["track.id"])
            af_ids[track_id] = line_idx
            ids[track_id] = 1


        for line_idx, line in enumerate(mf):
            mf_content.append(json.loads(line))
            track_id = int(mf_content[-1]["track.id"])
            mf_ids[track_id] = line_idx
            ids[track_id] = 1
        

        for track_id in ids:
            final_json_line = {}
            if track_id in af_ids:
                line_idx = af_ids[track_id]
                json_content = af_content[line_idx]
                
                for key in json_content:
                    final_json_line[key] = json_content[key]
            
            if track_id in mf_ids:
                line_idx = mf_ids[track_id]
                json_content = mf_content[line_idx]
                
                for key in json_content:
                    final_json_line[key] = json_content[key]

            
            out.write(final_json_line)
            
        




        
        




if __name__ == "__main__":
    main()
