README - Running inference.py

Requirements:

Model checkpoint files:

efficientnetb0_model.pt
dinov2-small.pt
deit-tiny-patch16-224.pt
resnet18_classification.pt

Image files:

1.jpg, 2.jpg, ..., 10.jpg

How to run:

Open a terminal or PowerShell in the folder containing the files and run: python inference.py
You will be given two options:

1. AUTO MODE
Runs all 4 models on all 10 images (1.jpg to 10.jpg)
At the end, a comparison table of predictions will be displayed and saved as model_comparison_output.txt.

2. MANUAL MODE
Allows you to:

Choose one model

Choose one image
It will then show predictions for that specific model-image pair.

Output:

Predictions printed in the console.

In AUTO MODE, a summary table is saved to model_comparison_output.txt.

