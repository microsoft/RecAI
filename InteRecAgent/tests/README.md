# Test Examples

The web integration tests run without model weights, catalog downloads, API keys
or third-party packages:

```bash
cd InteRecAgent
python -m unittest discover -s tests -p test_web_search.py -v
```

They exercise the production default resolver, plan-first agent and tool
dispatcher using a scripted model and a small catalog fixture. They check result
consumption, URLs, JSON/SSE MCP responses, request headers, disabled search,
explicit configuration, conversation isolation/reset, errors, timeouts and
interruption. They do not measure an LLM's ability to choose tools or run the
Gradio application with the full recommendation resources.

Here are some test examples/cases to test our Conversational Recommender System. Those examples are designed with different conditions:

- Intention Only
    - Hard conditions only
    - Soft conditions only
    - Mixed conditions
- Profile Only
    - User history only
    - User unwanted only
    - User history + unwanted
- Profile + Intention
    - User history + hard conditions
    - User history + soft conditions
    - User history + mixed conditions
    - User unwanted + hard conditions
    - User unwanted + soft conditions
    - User unwanted + mixed conditions
    - User history + unwanted + hard conditions
    - User history + unwanted + soft conditions
    - User history + unwanted + mixed conditions

All the cases consist of [single-turn conversations](./single_turn_cases.md) and multi-turn conversations.
