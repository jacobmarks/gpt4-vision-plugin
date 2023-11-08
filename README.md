## GPT-4 Vision Plugin

### Plugin Overview

On November 6, 2023, OpenAI made
[GPT-4 Vision](https://platform.openai.com/docs/guides/vision) available to
developers via an API. The model has the natural language capabilities of GPT-4,
as well as the (decent) ability to understand images. It can be prompted with
multimodal inputs, including text and a single image or multiple images.

This plugin allows you to integrate GPT-4 Vision natively into your AI and
computer vision workflows 💪!

## Installation

If you haven't already, install FiftyOne:

```shell
pip install fiftyone
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/jacobmarks/gpt4-vision-plugin
```

To use GPT-4 Vision, you will also need to set the environment variable
`OPENAI_API_KEY` with your API key. For information on estimating costs, see
[here](https://platform.openai.com/docs/guides/vision/calculating-costs)

### Getting your Data into FiftyOne

To use GPT-4 Vision, you will need to have a dataset of images in FiftyOne. If
you don't have a dataset, you can create one from a directory of images:

```python
import fiftyone as fo

dataset = fo.Dataset.from_images_dir("/path/to/images")

## optionally name the dataset and persist to disk
dataset.name = "my-dataset"
dataset.persistent = True

## view the dataset in the App
session = fo.launch_app(dataset)
```

## Operators

### `query_gpt4_vision`

Inputs:

- `query_text`: The text to prompt GPT-4 Vision with
- `max_tokens`: The maximum number of tokens to generate

The plugin's execution context will take all currently selected samples, encode
them, and pass them to GPT-4 Vision. The plugin will then output the response
from GPT-4 Vision 😄

Happy exploring!
