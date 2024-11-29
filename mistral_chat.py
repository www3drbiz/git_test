from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
import os
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# API 키 가져오기

api_key = "sM5AM2CWw2X1C33oQQSjHy4WsTjSZm81"

# Mistral AI 모델 초기화
chat = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.5,
    max_retries=2,
    api_key=api_key,
)

# 메시지 정의
message = [
    SystemMessage(
        "You are a helpful assistant that have expert knowledge about geography. Answer to the user sentence in {country}."
    ),
    AIMessage(
        "Hello, I'm a geography expert. My name is {name}!"
    ),
    HumanMessage(
        "How far is the {ACity} from the {BCity}? and also, how many hours does it take to get there?"
    )
    
]

template = ChatPromptTemplate.from_messages(
    [("system", "You are a list generating machine. Everything you are asked will be answered as a comma seperate list of max {max_item}. Do NOT reply with anything else" ),
      ("human", "{question}")]
)

prompt = template.format_messages(
    max_item=5,
    question="What are the top 5 countries with the highest GDP?"
)

response = chat.invoke(prompt)

print(response.content.strip().split(",")[:5])
