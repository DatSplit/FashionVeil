from PIL import Image
import matplotlib.pyplot as plt

# Define target size for all images
target_size = (600, 600)  # width, height in pixels

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    img = Image.open(f'missclassifications_examples/image_{i+1}.png')
    # Resize image to target size
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    ax.imshow(img_resized)
    ax.axis('off')

plt.tight_layout()
plt.savefig('combined_image.pdf', dpi=300, bbox_inches='tight')
