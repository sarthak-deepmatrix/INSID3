"""Segment target image(s) using reference image(s) and mask(s) with INSID3."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from datetime import datetime

import opts
from models import build_insid3_from_args


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_image(image_path: str) -> Image.Image:
    """Load and return PIL Image."""
    img = Image.open(image_path).convert('RGB')
    return img


def load_mask(mask_path: str) -> torch.Tensor:
    """Load mask and return as tensor."""
    mask = Image.open(mask_path).convert('L')
    mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
    return mask_tensor


def list_images(dir_path: str) -> list[str]:
    """Return sorted list of image file paths in a directory."""
    paths = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def match_files_by_stem(dir_a: str, dir_b: str) -> list[tuple[str, str]]:
    """Pair files from two directories by filename stem (ignoring extension).

    Returns sorted list of (path_in_a, path_in_b) tuples for stems present
    in both directories. Stems present in only one directory are skipped
    with a warning.
    """
    files_a = {Path(f).stem: f for f in os.listdir(dir_a) if Path(f).suffix.lower() in IMAGE_EXTENSIONS}
    files_b = {Path(f).stem: f for f in os.listdir(dir_b) if Path(f).suffix.lower() in IMAGE_EXTENSIONS}

    common_stems = sorted(set(files_a) & set(files_b))
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))

    for stem in only_a:
        print(f"Warning: '{files_a[stem]}' in {dir_a} has no matching mask, skipping")
    for stem in only_b:
        print(f"Warning: '{files_b[stem]}' in {dir_b} has no matching image, skipping")

    return [(os.path.join(dir_a, files_a[s]), os.path.join(dir_b, files_b[s])) for s in common_stems]


def save_segmentation_results(pred_mask, target_image, output_dir: str, name: str | None = None):
    """Save predicted mask as visualization and raw mask."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert mask to 0-255
    mask_np = (pred_mask.cpu().numpy() * 255).astype('uint8')

    # Resize mask to match original image size
    mask_pil = Image.fromarray(mask_np, mode='L')
    mask_pil = mask_pil.resize(target_image.size, Image.NEAREST)


    nested_dir = os.path.join(output_dir, name or str(datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(nested_dir, exist_ok=True)
    mask_pil.save(os.path.join(nested_dir, 'segmentation_mask.png'))
    print(f"Saved binary mask -> {os.path.join(nested_dir, 'segmentation_mask.png')}")

    # Create colored overlay visualization
    overlay = Image.new('RGBA', target_image.size, (0, 255, 0, 128))
    translucent_mask = mask_pil.point(lambda p: int(p * 0.5))
    overlay.putalpha(translucent_mask)

    image_rgb = target_image.convert('RGBA')
    result = Image.alpha_composite(image_rgb, overlay)
    result = result.convert('RGB')
    result.save(os.path.join(nested_dir, 'segmentation_overlay.png'))
    print(f"Saved overlay -> {os.path.join(nested_dir, 'segmentation_overlay.png')}")


def main(args: argparse.Namespace):
    print(args)

    # ──────── Resolve reference pairs ────────
    using_ref_dirs = bool(args.ref_images_dir or args.ref_masks_dir)
    using_ref_files = bool(args.ref_image or args.ref_mask)
    if using_ref_dirs == using_ref_files:
        raise ValueError(
            'Specify exactly one of (--ref-image and --ref-mask) or '
            '(--ref-images-dir and --ref-masks-dir).'
        )
    if using_ref_dirs:
        if not (args.ref_images_dir and args.ref_masks_dir):
            raise ValueError('--ref-images-dir and --ref-masks-dir must be given together.')
        ref_pairs = match_files_by_stem(args.ref_images_dir, args.ref_masks_dir)
        if not ref_pairs:
            raise ValueError(f'No matching reference image/mask pairs found in '
                              f'{args.ref_images_dir} and {args.ref_masks_dir}.')
    else:
        if not (args.ref_image and args.ref_mask):
            raise ValueError('--ref-image and --ref-mask must be given together.')
        ref_pairs = [(args.ref_image, args.ref_mask)]

    print(f'Using {len(ref_pairs)} reference pair(s):')
    for ref_image_path, ref_mask_path in ref_pairs:
        print(f'  image={ref_image_path}  mask={ref_mask_path}')

    # ──────── Resolve target images ────────
    using_target_dir = bool(args.target_dir)
    using_target_file = bool(args.target_image)
    if using_target_dir == using_target_file:
        raise ValueError('Specify exactly one of --target-image or --target-dir.')
    if using_target_dir:
        target_paths = list_images(args.target_dir)
        if not target_paths:
            raise ValueError(f'No target images found in {args.target_dir}.')
    else:
        target_paths = [args.target_image]

    print(f'Found {len(target_paths)} target image(s):')
    for target_path in target_paths:
        print(f'  {target_path}')

    # ──────── Model setup ────────
    model = build_insid3_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    # ──────── Load references (shared across all targets) ────────
    references = [(load_image(img_path), load_mask(mask_path)) for img_path, mask_path in ref_pairs]

    ref_label = Path(args.ref_images_dir).name if using_ref_dirs else Path(args.ref_image).stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ──────── Log run config ────────
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, f'config_{ref_label}_{run_timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, default=str, indent=2)
    print(f"Saved run config -> {config_path}")

    # ──────── Inference ────────
    with torch.no_grad():
        for target_path in target_paths:
            print(f"Loading target image: {target_path}")
            target_image = load_image(target_path)

            # set_reference()/set_target() must be called before each segment()
            # call, since segment() resets the model's internal state.
            for ref_image, ref_mask in references:
                model.set_reference(ref_image, ref_mask)
            model.set_target(target_image)

            pred_mask = model.segment()

            name = f'{ref_label}_{run_timestamp}'
            if using_target_dir:
                name = f'{ref_label}_{Path(target_path).stem}_{run_timestamp}'
            save_segmentation_results(pred_mask, target_image, args.output_dir, name=name)

    print(f"\n✓ Segmentation complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'INSID3 in-context segmentation with reference mask',
        parents=[opts.get_args_parser()],
    )
    parser.add_argument(
        '--ref-image',
        default=None,
        help='Path to a single reference image (1-shot). Mutually exclusive with --ref-images-dir.',
    )
    parser.add_argument(
        '--ref-mask',
        default=None,
        help='Path to a single reference mask (1-shot). Mutually exclusive with --ref-masks-dir.',
    )
    parser.add_argument(
        '--ref-images-dir',
        default=None,
        help='Directory of reference images (k-shot). Paired with --ref-masks-dir by filename stem.',
    )
    parser.add_argument(
        '--ref-masks-dir',
        default=None,
        help='Directory of reference masks (k-shot). Paired with --ref-images-dir by filename stem.',
    )
    parser.add_argument(
        '--target-image',
        default=None,
        help='Path to a single target image to segment. Mutually exclusive with --target-dir.',
    )
    parser.add_argument(
        '--target-dir',
        default=None,
        help='Directory of target images to segment using the same reference(s).',
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir or './output'

    main(args)
