from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv

import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="loops through 1-10")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Will detect the api key by its name in the .env file
llm = OpenAI()

code_prompt = PromptTemplate(
  template="Write a short {language} function that will {task}",
  input_variables=["language", "task"]
)

second_prompt = PromptTemplate(
  input_variables=["language", "code"],
  template="write a test for the following {language} code:\n {code}"
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  # rename output from 'text' to 'code'
  output_key="code"
)

second_chain = LLMChain(
  llm=llm,
  prompt=second_prompt,
  output_key="test",
)

chain = SequentialChain(
  chains=[code_chain, second_chain],
  input_variables=["task", "language"],
  output_variables=["test", "code"],
)

result = chain({
  "language": args.language,
  "task": args.task
})

print(result)