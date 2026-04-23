#!/usr/bin/env python3
"""Case 2 CNN wrapper: AugPred-pretrained frozen CNN ResNet + linear probe."""

from train_linear_probe_cnn import main


if __name__ == "__main__":
    main(default_config="case2_probe_cnn")
