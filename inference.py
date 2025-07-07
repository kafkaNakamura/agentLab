import openai
import time, tiktoken
from openai import OpenAI
import os, anthropic, json
import google.generativeai as genai
from openrouter_inference import query_openrouter_model
import time
from threading import Lock

# Rate limiting variables
RATE_LIMIT = 20  # 20 requests per minute
RATE_LIMIT_PERIOD = 60  # 60 seconds
request_times = []
rate_limit_lock = Lock()

def enforce_rate_limit():
    """Enforce rate limit of 20 requests per minute."""
    global request_times
    with rate_limit_lock:
        current_time = time.time()
        # Remove requests older than 1 minute
        request_times = [t for t in request_times if current_time - t <= RATE_LIMIT_PERIOD]
        
        # If we've hit the limit, sleep until the oldest request is more than 1 minute old
        if len(request_times) >= RATE_LIMIT:
            oldest_time = request_times[0]
            sleep_time = RATE_LIMIT_PERIOD - (current_time - oldest_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Add the current request time
        request_times.append(time.time())

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
        "llama4-scout": 0.0001 / 1000000,
        "gemini-2.0-flash": 0.0000 / 1000000,  
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
        "llama4-scout": 0.0001 / 1000000,
        "gemini-2.0-flash": 0.0000 / 1000000, 
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,  anthropic_api_key=None, groq_api_key = None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    # enforce_rate_limit()
    preloaded_api = os.getenv('OPENAI_API_KEY')
    preloaded_openrouter_api = os.getenv('OPENROUTER_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    if groq_api_key is not None:
        os.environ["GROQ_API_KEY"] = groq_api_key
    for _ in range(tries):
        try:
            answer = None
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "gemini-1.5-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str in ["llama4-scout", "llama-4-scout", "llama4-maverick", "llama-4-maverick"]:
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                
                # Determine which Groq model to use
                groq_model_map = {
                    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "llama-4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
                }
                
                completion = client.chat.completions.create(
                    model=groq_model_map[model_str],  # Dynamically select model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp if temp is not None else 0.7  # Default temp
                )
                answer = completion.choices[0].message.content
            elif model_str in ["llama3-70b-versatile", "llama-3.3-70b-versatile"]:
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp if temp is not None else 0.7
                )
                answer = completion.choices[0].message.content
            elif model_str in ["gemini-2.0-flash", "gemini2-flash"]:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)  # Uses gemini_api_key passed to query_model()
                model = genai.GenerativeModel('gemini-2.0-flash')
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                answer = model.generate_content(full_prompt).text
            elif model_str in ["gemini-2.5-flash", "gemini2.5-flash"]:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)  # Uses gemini_api_key passed to query_model()
                model = genai.GenerativeModel('gemini-2.5-flash')
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                answer = model.generate_content(full_prompt).text
            # trying huggingface models
            elif model_str == "Dorna-Llama3-8B-Instruct":
                from transformers import pipeline
                pipe = pipeline("text-generation", model="PartAI/Dorna-Llama3-8B-Instruct")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                # Generate the response
                response = pipe(
                    messages,
                    temperature=temp if temp is not None else 0.7,
                    max_new_tokens=512  # You can adjust this as needed
                )
                # Extract the generated text
                answer = response[0]['generated_text']
            elif model_str.startswith("openrouter/"):
                openrouter_model = model_str[len("openrouter/"):]  # Remove the prefix
                answer = query_openrouter_model(
                    model_str=openrouter_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    openrouter_api_key=preloaded_openrouter_api,
                    temp=temp
                )
            try:
                if (
                    model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]
                    or model_str.startswith("openrouter/")
                    or "deepseek" in model_str
                ):
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                if answer is not None:
                    TOKENS_OUT[model_str] += len(encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost: print(f"Cost approximation has an error? {e}")
            if answer is not None:
                return answer
            else:
                raise Exception("No answer returned from model.")
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))