"""Video benchmark eval suite for omni embedding models.

Runs each model on the curated MVEB tasks.
"""

from __future__ import annotations

import argparse
import sys

import mteb
import torch

MVEB_TASKS = [
    # Any2AnyRetrieval (8)
    "AVMemeExamVA2TRetrieval",
    "AudioCapsAVVA2TRetrieval",
    "MSRVTTA2V",
    "MSRVTTAT2V",
    "MSVDT2VRetrieval",
    "VALOR32KVT2ARetrieval",
    "VATEXT2VARetrieval",
    "VATEXV2ARetrieval",
    # VideoCentricQA (2)
    "OmniVideoBenchVideoCentricQA",
    "WorldSense1MinVideoAudioCentricQA",
    # VideoClassification (3)
    "AVEDatasetClassification",
    "AVMemeVideoClassification",
    "Diving48Classification.V2",
    # VideoClustering (1)
    "MELDEmotionAudioVideoClustering",
    # VideoPairClassification (3)
    "MELDVPairClassification",
    "RAVDESSAVVAPairClassification",
    "VideoConPairClassification",
    # VideoZeroshotClassification (1)
    "MELDVideoZeroShot",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--fps", type=float, default=None,
                        help="If unset, falls back to the wrapper default (e.g. None for EBind).")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Override the default task list.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    model_kwargs = {}
    if args.fps is not None:
        model_kwargs["fps"] = args.fps
    if args.max_frames is not None:
        model_kwargs["max_frames"] = args.max_frames
    if args.num_frames is not None:
        model_kwargs["num_frames"] = args.num_frames

    model = mteb.get_model(args.model, **model_kwargs)

    task_names = args.tasks if args.tasks else list(MVEB_TASKS)

    tasks = mteb.get_tasks(tasks=task_names)
    print(f"Running {len(tasks)} tasks on {args.model} (fps={args.fps}, batch={args.batch_size})")

    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model,
        output_folder=args.output_folder,
        verbosity=2,
        raise_error=False,
        encode_kwargs={"batch_size": args.batch_size},
    )


if __name__ == "__main__":
    sys.exit(main())
