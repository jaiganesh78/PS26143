import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CandidateClassifier(nn.Module):

    def __init__(
        self,
        feature_dim=8,
        pretrained=True,
    ):
        super().__init__()

        encoder_weights = (
            "imagenet"
            if pretrained
            else None
        )

        self.encoder = smp.encoders.get_encoder(
            "resnet18",
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        encoder_dim = (
            self.encoder.out_channels[-1]
        )

        self.image_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                encoder_dim,
                256,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.feature_head = nn.Sequential(
            nn.Linear(
                feature_dim,
                64,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                256 + 64,
                128,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        image,
        features,
    ):

        features_maps = self.encoder(
            image
        )

        image_embedding = (
            self.image_head(
                features_maps[-1]
            )
        )

        metadata_embedding = (
            self.feature_head(
                features
            )
        )

        combined = torch.cat(
            [
                image_embedding,
                metadata_embedding,
            ],
            dim=1,
        )

        return self.classifier(
            combined
        )