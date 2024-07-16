from transformers import pipeline


def expand_prompt(prompt, max_length=50):

    generator = pipeline("text-generation", model="gpt2")
    responses = generator(prompt, max_length=max_length, num_return_sequences=1)

    return responses[0]["generated_text"]
