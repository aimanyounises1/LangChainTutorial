from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langsmith import Client

from reviewing.langgraph_review.tools import tavily_search

load_dotenv()

llm = init_chat_model(model='ollama:nemotron-3-nano:latest', temperature=0.1, profile={
    "tool_calling": True,
    "structured_output": True,
})

client = Client()
# This is a ChatPromptTemplate
RAG_PROMPT = client.pull_prompt("rlm/rag-prompt")


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    If context is available (from tool execution), use the RAG prompt to generate an answer.
    Otherwise, let the LLM decide (e.g., call a tool).
    """
    messages = state['messages']
    
    # Check if we have context from a previous tool call
    context = ""
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            context = message.content
            break
            
    if context:
        # We have context. Use .partial() to inject it into the prompt.
        # This creates a new prompt template that only needs 'question'.
        rag_prompt_with_context = RAG_PROMPT.partial(context=context)
        
        # Extract the user's original question
        question = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                question = message.content
                break
                
        if not question:
            question = messages[0].content

        # Create the chain with the partially filled prompt
        chain = rag_prompt_with_context | llm
        
        # Invoke the chain with the remaining variable
        response = chain.invoke({"question": question})
        
        return {'messages': [response]}
    
    else:
        # No context yet. Pass the messages to the LLM.
        # It will likely decide to call the 'tavily_search' tool.
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}


tools = [tavily_search]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)
