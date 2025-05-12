from huggingface_hub import InferenceClient

# Replace with your own token if needed
client = InferenceClient(api_key="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")

# Generate an image
image = client.text_to_image(
    prompt="a man stuck in side phone in a scary way",
    model="stabilityai/stable-diffusion-xl-base-1.0"
)

# Show the image
image.show()
