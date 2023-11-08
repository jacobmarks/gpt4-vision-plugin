"""GPT-4 Vision plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo
import fiftyone.core.utils as fou

import base64
import requests


def allows_openai_models():
    """Returns whether the current environment allows openai models."""
    return "OPENAI_API_KEY" in os.environ


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def query_gpt4v(ctx):
    """Queries the GPT-4 Vision model."""
    dataset = ctx.dataset
    sample_ids = ctx.selected
    query_text = ctx.params.get("query_text", None)
    max_tokens = ctx.params.get("max_tokens", 300)

    messages_content = []
    text_message = {"type": "text", "text": query_text}
    messages_content.append(text_message)
    for sample_id in sample_ids:
        filepath = dataset[sample_id].filepath
        base64_image = encode_image(filepath)
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        messages_content.append(image_message)

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{"role": "user", "content": messages_content}],
        "max_tokens": max_tokens,
    }

    api_key = os.environ.get("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    content = response.json()
    if type(content) == str:
        return content
    elif "error" in content:
        return content["error"]["message"]
    else:
        return content["choices"][0]["message"]["content"]


class QueryGPT4Vision(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="query_gpt4_vision",
            label="GPT-4 Vision: Chat with your images!",
            dynamic=True,
        )
        _config.icon = "/assets/icon_light.svg"
        _config.dark_icon = "/assets/icon_dark.svg"
        _config.light_icon = "/assets/icon_light.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="GPT-4 Vision",
                icon="/assets/icon_dark.svg",
                dark_icon="/assets/icon_dark.svg",
                light_icon="/assets/icon_light.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="GPT-4 Vision",
            description="Ask a question about the selected image!",
        )

        openai_flag = allows_openai_models()
        if not openai_flag:
            inputs.message(
                "no_openai_key",
                label="No OpenAI API Key. Please set OPENAI_API_KEY in your environment.",
            )
            return types.Property(inputs)

        num_selected = len(ctx.selected)
        if num_selected == 0:
            inputs.str(
                "no_sample_warning",
                view=types.Warning(
                    label=f"You must select a sample to use this operator"
                ),
            )
        else:
            if num_selected > 10:
                inputs.str(
                    "many_samples_warning",
                    view=types.Warning(
                        label=(
                            f"You have {num_selected} samples selected. GPT-4 Vision"
                            " charges per image. Are you sure you want to continue?"
                        )
                    ),
                )

            inputs.str(
                "query_text", label="Query about your images", required=True
            )

        inputs.int(
            "max_tokens",
            label="Max tokens",
            default=300,
            description="The maximum number of tokens to generate. Up to 2048.",
            view=types.FieldView(),
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        question = ctx.params.get("query_text", None)
        answer = query_gpt4v(ctx)
        return {"question": question, "answer": answer}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("question", label="Question")
        outputs.str("answer", label="Answer")
        header = "GPT-4 Vision: Chat with your images"
        return types.Property(outputs, view=types.View(label=header))


def register(plugin):
    plugin.register(QueryGPT4Vision)
