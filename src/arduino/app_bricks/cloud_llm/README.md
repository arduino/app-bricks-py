# Cloud LLM brick

This directory contains the implementation of the Cloud LLM brick, which provides an interface to interact with cloud-based Large Language Models (LLMs) through their REST API.

## Overview

The Cloud LLM brick allows users to send prompts to a specified LLM service and receive generated responses.
It can be configured to work with a curated set of LLM providers that offer RESTful APIs, notably: ChatGPT, Claude and Gemini.

## Prerequisites

Before using the Cloud LLM brick, ensure you have the following:
- An account with a cloud-based LLM service (e.g., OpenAI, Cohere, etc.).
- API access credentials (API key or token) for the LLM service.
- Network connectivity to access the LLM service endpoint.

## Features

- Send prompts to a cloud-based LLM service.
- Receive and process responses from the LLM.
- Supports both one-shot requests and memory for follow-up questions and answers.
- Supports a curated set of LLM providers.

## Code example and usage
Here is a basic example of how to use the Cloud LLM brick:

```python
from arduino.app_bricks.cloud_llm import CloudLLM
from arduino.app_utils import App

llm = CloudLLM(api_key="your_api_key_here")

App.start_bricks()

response = llm.chat("What is the capital of France?")
print(response)

App.stop_bricks()
```
