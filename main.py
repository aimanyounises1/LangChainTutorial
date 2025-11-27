# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import asyncio
from langchain_ollama import  ChatOllama

async def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    model = ChatOllama(
        model="qwen3:30b-a3b",
        temperature=0.8,
        # other params ...
    )

    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    response = await model.ainvoke(input="What are literals in CPP?")
    print(response.content)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(print_hi('PyCharm'))
