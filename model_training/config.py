import os
from dotenv import load_dotenv

# Check if .env file exists in the current working directory
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(".env file loaded and environment variables set.")
else:
    print("No .env file found in the current directory.")

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN") 
