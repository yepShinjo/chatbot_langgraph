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

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classifiy if the message requires an emotional (therapist) or logical response."
    )

class State(TypedDict):
    # messages gonna be typed list, whenever we want to add messages use 'add_messages'
    messages: Annotated[list, add_messages]
    message_type: str | None


''' ALL of these 4 classes | classify_message | router | therapist_agent | logical_agent | are NODE'''
def classify_message(state: State):

    # whatever the user's last type we're gonna get that
    last_message = state["messages"][-1]

    classifier_llm = llm.with_structured_output(MessageClassifier)

    result =  classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classifiy the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role":"user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}

def router(state: State):
    # is the message considered either emotional or logical ? i could have said
    # state.get("emotional", "logical")
    # but if we want to add more type, we can do it like this, and if the message doesnt fall to any type category, then we fallback to it being 'logical'
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    # else
    return {"next": "logical"}

def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    # router is the source of this conditional branches
    "router",
    # now, look at the 'next' in the state.
    lambda state: state.get("next"),
    # if its therapist, then we go to therapist. if its logical, then go to logical
    # or i can just make it as, "apple": "therapist", "banana": "logical"
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("bye")
            break

        # if the messages is empty list (means its a fresh start), add the role and content to our state messages
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()


#---------------------------------------------------------

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:

    pass

#-----------------------------------------------------------