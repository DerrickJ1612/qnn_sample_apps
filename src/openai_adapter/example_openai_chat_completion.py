from openai import OpenAI
import json
from pprint import pprint

client = OpenAI(
    
    base_url="http://127.0.0.1:8000/",
    api_key="dummy"
)

json_format = {
  "routine_name": "Morning Flow",
  "difficulty": "beginner", 
  "total_duration": 15,
  "poses": [
    {
      "step": 1,
      "pose_name": "mountain_pose",
      "hold_duration": 30,
      "instructions": "Stand tall with feet hip-width apart...",
      "key_alignments": ["straight spine", "relaxed shoulders"],
      "max_attempts": 3
    },
    {
      "step": 2, 
      "pose_name": "downward_dog",
      "hold_duration": 45,
      "instructions": "From mountain pose, fold forward...",
      "transition_cue": "Slowly walk hands forward",
      "max_attempts": 3
    }
  ]
}

response = client.chat.completions.create(
    model="gemma-1b",
    messages=[
        {"role":"system", "content":f"You are a yoga instructor. You MUST respond ONLY with valid JSON in exactly this format: {json.dumps(json_format)}. Do not include any other text, explanations, or formatting. Return only the JSON object."},
        {"role":"user", "content":"Please provide a yoga routine to address shoulder pain"},
        # {"role":"assistant", "content":"For tight hamstrings let's begin with downward dog"}
        ],
    max_tokens=1000
)
pprint(response)