from m2t.train import train
import torch
import transformers
from m2t.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)

def add_my_arguments(parser):
    
    parser.add_argument("--local-rank", type=int, default=0)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    add_my_arguments(parser)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()
    train(args=args, model_args=model_args, training_args=training_args, data_args=data_args)
