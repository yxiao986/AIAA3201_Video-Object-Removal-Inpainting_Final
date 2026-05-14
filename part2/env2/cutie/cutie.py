import argparse
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


@torch.inference_mode()
@torch.cuda.amp.autocast()
def run(args):
    # Load the default pretrained Cutie model from the installed upstream repo.
    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = args.max_internal_size

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise ValueError(f"No frames found in {args.image_dir}")
    mask = Image.open(args.first_mask)
    palette = mask.getpalette()
    
    objects = np.unique(np.array(mask))
    objects = objects[objects != 0].tolist()  
    mask = torch.from_numpy(np.array(mask)).cuda()

    print(f"Starting Cutie propagation on {len(images)} frame(s); found {len(objects)} object id(s).")

    for ti, image_file in enumerate(images):
        image = Image.open(image_file)
        image = to_tensor(image).cuda().float()

        # The first frame initializes object identities; later frames are propagated automatically.
        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)
        else:
            output_prob = processor.step(image)

        out_mask = processor.output_prob_to_mask(output_prob)

        # Preserve the original palette when available so object ids remain visually consistent.
        out_mask_img = Image.fromarray(out_mask.cpu().numpy().astype(np.uint8))
        if palette is not None:
            out_mask_img.putpalette(palette)

        save_path = args.output_dir / f"{image_file.stem}{args.output_ext}"
        if args.rgb_output:
            out_mask_img.convert("RGB").save(save_path)
        else:
            out_mask_img.save(save_path)
        
        if ti % 10 == 0:
            print(f"Processed: {ti}/{len(images)} -> {save_path.name}")
            
    print(f"Done. Masks saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Propagate a first-frame mask with Cutie.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--first_mask", type=Path, required=True, help="Mask image for the first frame.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for propagated masks.")
    parser.add_argument("--max_internal_size", type=int, default=480, help="Cutie internal size; use -1 for original size.")
    parser.add_argument("--output_ext", type=str, default=".png", choices=[".png", ".jpg"], help="Output mask extension.")
    parser.add_argument("--rgb_output", action="store_true", help="Save masks as RGB images for pipelines requiring 3-channel masks.")
    run(parser.parse_args())

if __name__ == '__main__':
    main()
