

from azure.identity import get_bearer_token_provider, AzureCliCredential  
from openai import AzureOpenAI 
cost = 0
credential = AzureCliCredential() 
token_provider = get_bearer_token_provider( 
        credential, 
        "https://cognitiveservices.azure.com/.default") 
client = AzureOpenAI( 
        azure_endpoint="https://humanoidai.openai.azure.com/", 
        azure_ad_token_provider=token_provider, 
        api_version="2024-02-15-preview", 
        max_retries=1, )




response = client.chat.completions.create(
            model="gpt-35-turbo",
            max_tokens=1, 
            messages=[{"role":"user", "content":"no"}]  
        )

result = response.choices[0].message.content
prompt_tokens = response.usage
print(type(prompt_tokens.completion_tokens))
a = 2
a += prompt_tokens.completion_tokens+1
print(a)


# response = client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input = ["apple"]
#             )


# result = response.data[0].embedding
# print(result)

# prompt_tokens = response.usage
# print(prompt_tokens)
# print(prompt_tokens.prompt_tokens)
# print(prompt_tokens.completion_tokens)