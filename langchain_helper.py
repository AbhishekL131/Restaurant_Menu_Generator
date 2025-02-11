from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from secret_key import gemini_api_key
import google.generativeai as genai
import os

os.environ["GOOGLE_API_KEY"] = gemini_api_key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

def generate_restaurant_name(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest only one fancy name."
    )

    name_chain = LLMChain(llm=model, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest me some food menu items for {restaurant_name}. Return them as a comma-separated list."
    )

    food_items_chain = LLMChain(llm=model, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    response = chain({'cuisine': cuisine})
    return response  # No need to import response from main.py
