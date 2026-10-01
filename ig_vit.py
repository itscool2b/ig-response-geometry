"""Optional primitive demo. Historical source is preserved under legacy/2026-09-30."""
from demo_cli import parse_demo_args

def main():
    args = parse_demo_args("vision")
    #imports
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from integrated_gradients import integrated_gradients
    from overlays import model_input_rgb, save_new_figure

    #model
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights).to(args.device)
    model.train(False)
    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    #preprocess
    img = Image.open(args.image).convert('RGB')
    preprocess = weights.transforms()
    input_tensor = preprocess(img).unsqueeze(0).to(args.device)

    #predicted class
    with torch.no_grad():
        logits = model(input_tensor).squeeze(0)
        class_id = logits.argmax().item()
        class_name = weights.meta["categories"][class_id]
        prob = torch.softmax(logits, dim=0)[class_id].item() * 100
        print(f"predicted: {class_name} (logit={logits[class_id].item():.2f}, {prob:.1f}%)")

    #forward_fn â€” log_softmax of predicted class
    def forward_fn(x):
        logits = model(x).squeeze(0)
        return torch.log_softmax(logits, dim=-1)[class_id]

    #ig â€” black image baseline (zeros in pixel space, preprocessed), m=300
    black_img = Image.new('RGB', (224, 224), (0, 0, 0))
    baseline = preprocess(black_img).unsqueeze(0).to(args.device)
    attr = integrated_gradients(forward_fn, input_tensor, baseline, m=args.m)

    #visualize
    attr_map = attr.squeeze(0).sum(dim=0).abs().detach().cpu().numpy()
    attr_map = attr_map / attr_map.max() if attr_map.max() else attr_map

    original = model_input_rgb(input_tensor, preprocess.mean, preprocess.std)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"ViT-B/16  |  predicted: {class_name}  ({prob:.1f}%)", fontsize=13, fontweight="bold", y=1.02)

    ax1.imshow(original)
    ax1.set_title("model input after resize/crop", fontsize=11)
    ax1.axis("off")

    im2 = ax2.imshow(attr_map, cmap="hot")
    ax2.set_title("pixel-coordinate IG (patch-structured model)", fontsize=11)
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="normalized attribution")

    ax3.imshow(original)
    ax3.imshow(attr_map, cmap="hot", alpha=0.5)
    ax3.set_title(f"overlay ({class_name})", fontsize=11)
    ax3.axis("off")

    fig.text(0.5, -0.02,
             f"IG params: m={args.m} intervals, trapezoid  |  baseline: black image (preprocessed zeros)  |  target: log_softmax  |  note: patch-shaped patterns from 16px patch embedding",
             ha="center", fontsize=9, fontstyle="italic", color="0.4")

    plt.tight_layout()
    save_new_figure(fig, args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

    #cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
