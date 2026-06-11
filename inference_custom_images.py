import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import opts
from models import build_insid3_from_args

def load_image(image_path, device):
    """Loads an image and converts it to a tensor expected by the model."""
    img = Image.open(image_path).convert('RGB')
    # Convert to tensor and add batch dimension: [1, C, H, W]
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    return img_tensor

def main(args):
    print("Loading model...")
    model = build_insid3_from_args(args)
    model.to(args.device)
    model.eval()

    print(f"Loading reference image: {args.ref_image}")
    src_img = load_image(args.ref_image, args.device)
    
    print(f"Loading target image: {args.target_image}")
    trg_img = load_image(args.target_image, args.device)

    # Parse source keypoints from the command line argument (e.g., "50,50 100,120")
    points = []
    for pt in args.src_points.split():
        x, y = map(float, pt.split(','))
        points.append([x, y])
    
    # Format shape to [Batch=1, Num_Points, 2]
    src_kps = torch.tensor(points, dtype=torch.float32).to(args.device)

    print(f"Finding matches for points: {points}...")
    with torch.no_grad():
        model.set_reference(src_img)
        model.set_target(trg_img)
        # Get the predicted keypoints on the target image
        pred_kps = model.match(src_kps, use_debiased=args.debiased)

    print("\n--- Results ---")
    for i, (src_pt, pred_pt) in enumerate(zip(src_kps, pred_kps)):
        print(f"Point {i+1} | Source (x,y): [{src_pt[0]:.1f}, {src_pt[1]:.1f}] --> Target (x,y): [{pred_pt[0]:.1f}, {pred_pt[1]:.1f}]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('INSID3 custom image semantic correspondence', parents=[opts.get_args_parser()])
    
    # Add custom arguments for ad-hoc inference
    parser.add_argument('--ref-image', type=str, required=True, help='Path to source/reference image')
    parser.add_argument('--target-image', type=str, required=True, help='Path to target image')
    parser.add_argument('--src-points', type=str, required=True, help='Source points to match in "X,Y X,Y" format')
    parser.add_argument('--debiased', action='store_true', help='Use positional debiased features')
    
    args = parser.parse_args()
    main(args)