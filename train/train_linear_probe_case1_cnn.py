#!/usr/bin/env python3
"""Case 1 CNN wrapper: randomly initialized frozen CNN ResNet + linear probe."""

from train_linear_probe_cnn import main


if __name__ == "__main__":
    main(default_config="case1_probe_cnn")
