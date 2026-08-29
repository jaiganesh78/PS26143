import segmentation_models_pytorch as smp


def build_model(
    architecture="unet",
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=2,
    classes=1,
):
    """
    Build the PS26143 baseline segmentation model.

    Architecture:
        U-Net

    Encoder:
        ResNet34

    Input:
        2 channels (VV + VH)

    Output:
        1 binary segmentation channel
    """

    architecture = architecture.lower()

    if architecture != "unet":
        raise ValueError(
            f"Unsupported architecture "
            f"for baseline: {architecture}"
        )

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

    return model