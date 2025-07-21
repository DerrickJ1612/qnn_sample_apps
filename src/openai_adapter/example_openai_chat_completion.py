from openai import OpenAI


client = OpenAI(
    
    base_url="http://127.0.0.1:8000/",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gemma-1b",
    messages=[
        {"role":"system", "content":"You are a very experienced yoga instructor"},
        {"role":"user", "content":"Please provide a yoga routine to address tight hamstrings"},
        {"role":"assistant", "content":"For tight hamstrings let's begin with downward dog"}
        ],
    max_tokens=100
)
print(response)