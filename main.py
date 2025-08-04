from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model(
    'gpt-4o'
)

class State(TypedDict):
    # messages gonna be typed list, whenever we want to add messages use 'add_messages'
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# make nodes
# for This node we take the current state
# state is basically a bunch of messages

# this line practically implement memory for us, 
# (behind every message pass on the memory from the state, and wrap em up in a list)
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# add_node("node_name", node)
graph_builder.add_node("chatbot", chatbot)

# STARD and END needed to define the start and end of the graph
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("enter a message: ")

# graph.invoke will take 'messages' from class State above
state = graph.invoke({"messages": [{"role": "user", "content":user_input}]})
print(state["messages"])
print(state["messages"][-1].content)

#---------------------------------------------------------

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:

    pass

#-----------------------------------------------------------