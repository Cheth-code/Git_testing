from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from langsmith import traceable

from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o-mini")

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # raw email data
    email_content : str
    sender_email : str
    email_id : str

    # classification result
    classification: EmailClassification | None

    # API results
    search_results: list[str] | None # list of raw documents 
    customer_history: dict | None # raw customer data

    # generate content
    draft_response: str | None
    messages: list[str] | None

@traceable(run_type="chain")
def read_email(state: EmailAgentState) -> dict:
    """ Extract and parse email content"""
    return{
        "messages": [HumanMessage(content=f"Porcessing email: {state['email_content']}")]
    }

@traceable(run_type="chain")
def classify_user_intent(state: EmailAgentState) -> Command[Literal ["search_documentation", "human_review", "draft_response", "bug_tracking"]] :
    """ Use of an LLM to classify the email intent and urgency, then route accordingly """

    # creating a structured LLM response
    structured_llm = llm.with_structured_output(EmailClassification) # as we are classifying the EmailClassification state is passed

    # Format the prompt on-demand
    classification_prompt = f"""
    Analyze the customer's email and classify them
    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic and summary.

    """
    # the route where if else is used more often
    classification = structured_llm.invoke(classification_prompt)
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification ['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification ['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(
        update={"classification": classification},
        goto=goto
    )                               

@traceable(run_type="chain")
def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """ Search knowledge base for relevant information """

    # build a search query form classifcation
    classification = state.get('classification', {})
    query = f"{classification.get('intent','')} {classification.get('topic', '')}"
    
    try:
        # search algo herer
        # store raw search results
        search_results = [
            "Reset pass word via settings > security > change password",
            "Password must be at least 12 chars",
            "Include uppercase, lowercase, numbers and symbols"

        ]
    except SearchAPIError as e:
        # for recoverable search error, store error and continue
        search_results = [f"Search not available: {str(e)}"]

    return Command(
        update={"search_results":search_results},
        goto="draft_response"
    )       
 
@traceable(run_type="chain")
def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """ Create or update a bug """

    # Create ticket in bug tracking system
    ticket_id = "BUG-1234"

    return Command(
        update= {
            "search_results": [f"Bug Ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )

@traceable(run_type="chain")
def draft_response(state: EmailAgentState) -> Command[Literal["human_review"]]:
    """ Generate response using context and route based on quality """
    classification = state.get('classification', {})

    # format context from raw state 
    context_section = []

    if state.get('search_results'):
        # format search results for the prompt
        format_docs = "\n".join([f"-{doc}" for doc in state['search_results']])
        context_section.append(f"Relevant documentation: \n {format_docs}")

    if state.get('customer_history'):
        # format customer data for the prompt
        context_section.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

    draft_prompt = f"""
        Draft a response to this customer's email
        {state['email_content']}

        Email intent: {classification.get('intent','unkown')}
        Urgency level: {classification.get('urgency','medium')}

        {chr(10).join(context_section)}

        Guidelines:
        - Be professional and helpful
        - Address their specific concern
        - Use the provided documentation when relevant
    """           
    response = llm.invoke(draft_prompt)

    # Determine if human review needed based on urgency and intent
    needs_review = (
        classification.get('urgency') in ['high','critical'] or classification.get('intent') == 'complex'
     )     
    
    goto = "human_review" if needs_review else "send_reply"

    return Command(update={"draft_response": response.content},
                   goto=goto
                   )

@traceable(run_type="chain")
def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """ pause for human review using and interrupt and route based on decision """

    classification = state.get('classification',{})

    # interrput() must come and code before it will re-run on resume
    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content', ''),
        "draft_response":state.get('draft_response',''),
        "urgency": classification.get("urgency"),
        "intent": classification.get("intent"),
        "action": "Please review and approve/edit this response"
    })

    # Now process the human's decision
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response', ''))},
            goto="send_reply"
        )
    else:
        # if rejected human will handle directly
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """ Send the email Response """
    print(f"Sending the reply: {state['draft_response'] [:100]}")
    return {}

graph_builder=StateGraph(EmailAgentState)

graph_builder.add_node("read_email",read_email)
graph_builder.add_node("classify_intent",classify_user_intent)

# Add retry policy for nodes that might have transient failures
graph_builder.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
graph_builder.add_node("bug_tracking",bug_tracking)
graph_builder.add_node("human_review",human_review)
graph_builder.add_node("draft_response",draft_response)
graph_builder.add_node("send_reply",send_reply)

# to add edges
graph_builder.add_edge(START,"read_email")
graph_builder.add_edge("read_email","classify_intent")
graph_builder.add_edge("send_reply", END)

# compile with checkpointer for perisitence
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# Test with an urgent billing issue
initial_state= {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

# Run with thread_id for persistance
config = {"configurable": {"thread_id": "customer@123"}}
result = app.invoke(initial_state, config)

print(f"human review interuppt: {result['__interrupt__']}")

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund..."
    }
)

final_result = app.invoke(human_response, config)
print(f"Email sent successfully!")