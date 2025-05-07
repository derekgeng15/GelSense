import torch
import torch.nn as nn
from torchvision.models import vgg16

class GraspInf(nn.Module):
    def __init__(self,
                 mlp_hidden_dim: int = 512,
                 pretrained: bool = True):
        """
        Args:
            mlp_hidden_dim: hidden size of the fusion MLP
            pretrained: whether to initialize VGG16 from ImageNet weights
        """
        super().__init__()

        # Vision streams (separate weights)
        self.vis_a = vgg16(pretrained=pretrained)
        self.vis_b = vgg16(pretrained=pretrained)
        # Truncate classifier to produce 4096-dim features
        self.vis_a.classifier = nn.Sequential(*list(self.vis_a.classifier.children())[:-1])
        self.vis_b.classifier = nn.Sequential(*list(self.vis_b.classifier.children())[:-1])

        # Touch stream (shared weights)
        self.touch = vgg16(pretrained=pretrained)
        self.touch.classifier = nn.Sequential(*list(self.touch.classifier.children())[:-1])

        # Fusion MLP: 4 * 4096 -> mlp_hidden_dim -> 1
        fusion_input_dim = 4 * 4096
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self,
                vis_before: torch.Tensor,
                vis_after:  torch.Tensor,
                delta_l: torch.Tensor,
                delta_r: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            vis_before: [B,224,224,3] RGB "before grasp" images
            vis_after : [B,224,224,3] RGB "at grasp" images
            delta_l: [B,224,224,3] left GelSight deformation
            delta_r: [B,224,224,3] right GelSight deformation

        Returns:
            [B] probability of grasp success
        """
        # Permute dimensions to [B, 3, 224, 224]
        vis_before = vis_before.permute(0, 3, 1, 2)
        vis_after = vis_after.permute(0, 3, 1, 2)
        delta_l = delta_l.permute(0, 3, 1, 2)
        delta_r = delta_r.permute(0, 3, 1, 2)

        # Vision feature extraction
        f_va = self.vis_a(vis_before)  # [B,4096]
        f_vb = self.vis_b(vis_after)   # [B,4096]

        # Touch deformation streams
        f_tl = self.touch(delta_l)     # [B,4096]
        f_tr = self.touch(delta_r)     # [B,4096]

        # Concatenate and predict
        feats = torch.cat([f_va, f_vb, f_tl, f_tr], dim=1)  # [B,4*4096]
        prob = self.mlp(feats).squeeze(1)                    # [B]
        return prob

class GraspInf_GelOnly(nn.Module):
    def __init__(self,
                 mlp_hidden_dim: int = 512,
                 pretrained: bool = True):
        """
        Args:
            mlp_hidden_dim: hidden size of the fusion MLP
            pretrained: whether to initialize VGG16 from ImageNet weights
        """
        super().__init__()

        # Touch stream (shared weights)
        self.touch = vgg16(pretrained=pretrained)
        self.touch.classifier = nn.Sequential(*list(self.touch.classifier.children())[:-1])

        # Fusion MLP: 4 * 4096 -> mlp_hidden_dim -> 1
        fusion_input_dim = 2 * 4096
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self,
                delta_l: torch.Tensor,
                delta_r: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            delta_l: [B,224,224,3] left GelSight deformation
            delta_r: [B,224,224,3] right GelSight deformation

        Returns:
            [B] probability of grasp success
        """
        delta_l = delta_l.permute(0, 3, 1, 2)
        delta_r = delta_r.permute(0, 3, 1, 2)
        # Touch deformation streams
        f_tl = self.touch(delta_l)     # [B,4096]
        f_tr = self.touch(delta_r)     # [B,4096]

        # Concatenate and predict
        feats = torch.cat([f_tl, f_tr], dim=1)  # [B,2*4096]
        prob = self.mlp(feats).squeeze(1)                    # [B]
        return prob
