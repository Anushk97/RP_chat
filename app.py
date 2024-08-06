from beam import App, Runtime, Image, Volume, QueueDepthAutoscaler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# Define the Beam app
app = App(
    name="high-performance-inference",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.9",
            python_packages=[
                "transformers",
                'torch',
                'accelerate',
                'optimum',
                'auto-gptq',
            ],  # You can also add a path to a requirements.txt instead
        ),
    ),
    # Storage Volume for model weights
    volumes=[Volume(name="shared-weights", path="./weights")],
)

# Autoscale by queue depth, up to 5 replicas
autoscaler = QueueDepthAutoscaler(max_tasks_per_replica=30, max_replicas=5)

# This function runs once when the container boots
def load_models():
    model_name_or_path = "TheBloke/Silicon-Maid-7B-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer

# Rest API initialized with loader and autoscaler
@app.rest_api(loader=load_models, autoscaler=autoscaler)
def predict(**inputs):
    # Retrieve cached model from loader
    model, tokenizer = inputs["context"]

    try:
        prompt = inputs["prompt"]
    # Use a default prompt if none is provided
    except KeyError:
        prompt = "Q: What is the largest animal?\nA:"

    # Generate response
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    response = generator(prompt, max_new_tokens=200)
    result = response[0]['generated_text'].strip()

    print(result)

    return {"prediction": result}

if __name__ == "__main__":
    app.run()