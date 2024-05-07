# Usage of RecLlama

To use RecLlama locally, you need to follow these steps:

1. Download the weights of RecLlama.

2. Install `fschat` package according to [FastChat](https://github.com/lm-sys/FastChat?tab=readme-ov-file#install)

3. Deploy the `RecLlama` model with OpenAI-Compatible RESTful APIs by running the following command:

    ```bash
    # First, launch the controller
    python3 -m fastchat.serve.controller
    # Then, launch the model worker(s)
    python3 -m fastchat.serve.model_worker --model-path path-to-recllama
    # Finally, launch the RESTful API server
    python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
    ```

4. Set the configuration for RecLlama API in `run.sh`

    ```bash
    API_KEY="EMPTY"
    API_BASE="http://localhost:8000/v1"
    API_VERSION="2023-03-15-preview"
    API_TYPE="open_ai"
    engine="RecLlama"   # model name
    bot_type="completetion" # model type
    ```

5. Run the `run.sh` script to start the chatbot.