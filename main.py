from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Will detect the api key by its name in the .env file
llm = OpenAI()

code_prompt = PromptTemplate(
  template="Write a short {language} function that will {task}",
  input_variables=["language", "task"]
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  # rename output from 'text' to 'code'
  output_key="code"
)

result = code_chain({
  "language": args.language,
  "task": args.task
})

print(result)