import argparse
import pathlib
import time
import cv2
import torch
import torch.backends.cudnn as cudnn

from batch_face import RetinaFace
from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run L2CS gaze estimation on a saved video file.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to run model: cpu or gpu:0')
    parser.add_argument('--snapshot', type=str,
                        default='models/L2CSNet_gaze360.pkl',
                        help='Path to model snapshot')
    parser.add_argument('--arch', type=str, default='ResNet50',
                        help='Network architecture: ResNet18, ResNet34, etc.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Path to save the output video')
    return parser.parse_args()

def main():
    args = parse_args()
    cudnn.enabled = True

    # Load model
    gaze_pipeline = Pipeline(
        weights=pathlib.Path(args.snapshot),
        arch=args.arch,
        device=select_device(args.device, batch_size=1)
    )

    # Open video file
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run pipeline
            results = gaze_pipeline.step(frame)

            # Draw and write to output
            frame = render(frame, results)
            out.write(frame)

    cap.release()
    out.release()
    print(f"Finished. Output saved to: {args.output}")

if __name__ == '__main__':
    main()
