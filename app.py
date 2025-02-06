from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

app = Flask(__name__)

# Load model
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


@app.route('/generate_comic', methods=['POST'])
def generate_comic():
    try:
        # Get story details from the request
        data = request.get_json()
        story_parts = data.get('story_parts', {})

        # Generate images for each part of the story
        images = []
        for part, prompt in story_parts.items():
            result = pipe(prompt)
            if result.images:
                image = result.images[0]
                # Convert image to byte format to send in response
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                images.append(img_byte_arr)

        if not images:
            return jsonify({"error": "No images generated."}), 400

        # Return the first generated image (you can return multiple images too)
        return send_file(images[0], mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
