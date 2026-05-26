import torch
import torchreid
import numpy as np
import cv2
from torchvision import transforms


class ReIDEmbedder:
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # OSNet is purpose-built for cross-viewpoint re-identification
        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )

        # Load your fine-tuned weights if available
        if model_path:
            torchreid.utils.load_pretrained_weights(self.model, model_path)

        self.model = self.model.to(self.device).eval()

        # Standard re-ID preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return L2-normalised 512-d embedding for a single crop."""
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat   = self.model(tensor)
        feat   = feat / feat.norm(dim=1, keepdim=True)
        return feat.squeeze().cpu().numpy()

    def embed_batch(self, crops: list, batch_size: int = 32) -> np.ndarray:
        """Batch embed all crops — much faster on GPU."""
        all_embs = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            tensors = torch.stack([
                self.transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                for c in batch
            ]).to(self.device)
            with torch.no_grad():
                feats = self.model(tensors)
                feats = feats / feats.norm(dim=1, keepdim=True)
            all_embs.extend(feats.cpu().numpy())
        return np.array(all_embs)