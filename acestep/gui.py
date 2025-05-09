"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os
import click

from .ui.components import create_main_demo_ui
from .pipeline_ace_step import ACEStepPipeline
from .data_sampler import DataSampler
import requests


@click.command()
@click.option(
    "--checkpoint_path",
    type=str,
    default="",
    help="Path to the checkpoint directory. Downloads automatically if empty.",
)
@click.option(
    "--server_name",
    type=str,
    default="127.0.0.1",
    help="The server name to use for the Gradio app.",
)
@click.option(
    "--port", type=int, default=7865, help="The port to use for the Gradio app."
)
@click.option("--device_id", type=int, default=0, help="The CUDA device ID to use.")
@click.option(
    "--share",
    type=click.BOOL,
    default=False,
    help="Whether to create a public, shareable link for the Gradio app.",
)
@click.option(
    "--bf16",
    type=click.BOOL,
    default=True,
    help="Whether to use bfloat16 precision. Turn off if using MPS.",
)
@click.option(
    "--torch_compile", type=click.BOOL, default=False, help="Whether to use torch.compile."
)
def main(checkpoint_path, server_name, port, device_id, share, bf16, torch_compile):
    """
    Main function to launch the ACE Step pipeline demo.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
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
    demo.launch(server_name=server_name, server_port=port, share=share)


if __name__ == "__main__":
    main()
