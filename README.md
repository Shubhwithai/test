# Tweet Generator

Tweet Generator is a simple LLM application that uses Langchain and OpenAI. This project generates tweets on any topic given by the user.

## Installing libraries

```bash
!pip install -qU langchain langchain-openai langchain-google-genai
```

## Setting up API Keys

Store your API keys to access OpenAI and Google models.

- **Get OpenAI API key:** https://platform.openai.com/account/api-keys
- **Get Google API key:** https://aistudio.google.com

```python
import os

os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY" 
os.environ['GOOGLE_API_KEY']  = "YOUR_GOOGLE_API_KEY"
```

## Using LLMs to run a query

```python
from langchain_openai import ChatOpenAI

# Using GPT-3.5-turbo
gpt3_model = ChatOpenAI(model_name = "gpt-3.5-turbo-0125") 
response = gpt3_model.invoke("Explain transformers architecture.")
print(response.content)

# Generate tweets on India
response = gpt3_model.invoke("Generate 3 tweets on India")
print(response.content)

# Using GPT-4 (requires access)
gpt4_model = ChatOpenAI(model_name = "gpt-4o")
response = gpt4_model.invoke("Explain transformers architecture.")
print(response.content)
```

## Using Gemini Models

```python
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest")
response = gemini_model.invoke("Give me 3 tweets on World War 1")
print(response.content)
```

## Using Prompt Templates

```python
from langchain import PromptTemplate

tweet_template = "Give me {number} tweets on {topic}."
tweet_template_hindi = "Give me {number} tweets on {topic} in Hindi. Don't use hashtags. "

tweet_prompt = PromptTemplate(template = tweet_template, input_variables = ['number', 'topic'])
tweet_prompt_hindi = PromptTemplate(template = tweet_template_hindi, input_variables = ['number', 'topic'])

tweet_template.format(number = 7, topic = "Submarine") 
```

## Using LLM Chains

```python
from langchain import LLMChain

tweet_chain = tweet_prompt | gpt3_model

new_tweet_chain = tweet_prompt_hindi | gpt3_model

response = new_tweet_chain.invoke({"number" : 5, "topic" : "Wars in Middle East"})
print(response.content)

response = tweet_chain.invoke({"number" : 5, "topic" : "Wars in Middle East"})
print(response.content)

#Interactive Loop
while True:
  topic = input("Tweet topic: ")
  number = input("Number of tweets: ")
  response = tweet_chain.invoke({"number" : number, "topic" : topic})
  print(response.content)
```

## Prompting Techniques

### Few Shot Prompting

```python
from langchain_openai import ChatOpenAI

gpt4_model = ChatOpenAI(model_name = "gpt-4o") 

response = gpt4_model.invoke("Give me a question on Newton's Law of Motion")
print(response.content)

# ... more examples of few-shot prompting
```

### Persona-based Prompting

```python
response = gpt4_model.invoke("""
Give me a tweet on Elon Musk.
""")
print(response.content)

response = gpt4_model.invoke("""You are Shakespeare, a poet known for his unique writing style.

Give me a tweet on Elon Musk.
""")
print(response.content)

# ... more examples of persona-based prompting
```

### Chain of Thought Prompting

```python
response = gpt3_model.invoke("""
If 20 wet shirts take 5 hours to dry. How much time will it take to dry 100 shirts?
""")
print(response.content)

# ... more examples of chain-of-thought prompting
```

### Emotional Prompting

```python
response = gpt4_model.invoke("""
Give me robust python code to create a snake game.

Every time you give a wrong answer, a kitten dies. Give good answers and Save the Kittens.
""")
print(response.content)

# ... more examples of emotional prompting
```

## Experimenting with Models and Prompting

- Try different models like GPT-3.5-turbo, GPT-4, Gemini, or LLaMA.
- Experiment with different prompting techniques and settings.
- Be aware of potential biases in LLM outputs. 

This code provides a starting point for building your own tweet generator.  Feel free to explore different models, prompts, and techniques to customize your application.
