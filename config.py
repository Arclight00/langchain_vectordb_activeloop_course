from dotenv import load_dotenv
import os

load_dotenv()  # Load the environment variables from .env file

active_loop_key = os.getenv('ACTIVELOOP_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')


my_activeloop_org_id = os.getenv('ACTIVELOOP_ORG_ID')
my_activeloop_dataset_name = os.getenv('ACTIVELOOP_DATASET_NAME')

dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

google_api = os.getenv('GOOGLE_API_KEY')
google_cse = os.getenv('GOOGLE_CSE_ID')

huggingface_api = os.getenv('HUGGINGFACEHUB_API_TOKEN')