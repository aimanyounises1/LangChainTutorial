"""Quick test of simplified implementation - runs in ~30 seconds"""

print("Testing simplified deep research agent...")
print("=" * 60)

try:
    from deep_research_create_agent import create_research_agent
    from langchain_core.messages import HumanMessage
    
    print("✅ Import successful!")
    print("\nCreating agent...")
    
    agent = create_research_agent(use_checkpointer=False)
    print("✅ Agent created!")
    
    print("\nRunning quick test query...")
    test_query = "What are the top 3 benefits of AI agents?"
    
    result = agent.invoke({
        "messages": [HumanMessage(content=test_query)]
    }, {"configurable": {"thread_id": "test_123"}})
    
    print("✅ Query completed!")
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    
    if result and "messages" in result:
        final_msg = result["messages"][-1]
        if hasattr(final_msg, "content"):
            print(final_msg.content[:500] + "...")
            print("\n✅ SUCCESS! Simplified agent is working!")
        else:
            print("⚠️  No content in response")
    else:
        print("⚠️  No result returned")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you have:")
    print("  - langchain >= 1.0")
    print("  - langchain-community")
    print("  - langchain-ollama")
    print("  - langchain-tavily")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
