import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7865)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--share", type=bool, default=False)
parser.add_argument("--bf16", type=bool, default=True)
parser.add_argument("--torch_compile", type=bool, default=False)
args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)


from ui.components import create_main_demo_ui
from pipeline_ace_step import ACEStepPipeline
from data_sampler import DataSampler
import requests


def main(args):
    model_demo = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        torch_compile=args.torch_compile
    )
    data_sampler = DataSampler()

    def text2music(*args, **kwargs):
        try:
            requests.post("http://authproxy:7860/acestep/join", timeout=600)
            return model_demo(*args, **kwargs)
        finally:
            requests.post("http://authproxy:7860/acestep/leave", timeout=5)

    demo = create_main_demo_ui(
        text2music_process_func=text2music,
        sample_data_func=data_sampler.sample,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main(args)
