# Import base class for custom chat models in LangChain
from langchain_core.language_models.chat_models import BaseChatModel # you use this to customize your own llm

# Import message types used in LangChain conversations
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import structures for formatting the output of the chat model
from langchain_core.outputs import ChatGeneration, ChatResult

# Import the LiteLLM library for calling language models via API
import litellm


# Define a custom wrapper for LiteLLM that works with LangChain
class LiteLLMLangChainWrapper(BaseChatModel):
    
    # This method is required by LangChain and is called when generating a response
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult: # messages is a list of HumanMessage, AIMessage, SystemMessage;top is a list of strings that indicate when to stop generating text;run_manager is used for tracking the generation process;**kwargs allows passing additional parameters to the LLM API like temperature, max tokens, etc.
        # Initialize an empty list to collect formatted message strings
        prompt_parts = []

        # Loop through all messages in the conversation
        for msg in messages:
            # If the message is from the user
            if isinstance(msg, HumanMessage):
                prompt_parts.append(f"Human: {msg.content}")
            # If the message is from the AI
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"AI: {msg.content}")
            # If the message is a system-level instruction
            elif isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            # Raise an error for unsupported message types
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        # Join all formatted messages into one complete prompt string
        prompt = "\n".join(prompt_parts)

        # Call the LiteLLM API with the prompt
        response = litellm.completion(
            model="together_ai/mistralai/Mistral-7B-Instruct-v0.2",  # Specify which LLM to use
            messages=[{"role": "user", "content": prompt}],          # Provide the full prompt as a single user message
            api_key="309ce89cad6d001f10acf44e5e5bed429ab3cea28b0a5167de6e961e7c97f7eb"  # API key to authenticate with LiteLLM (should ideally be stored securely)
        )

        # Extract the generated content from the LLM's response
        content = response["choices"][0]["message"]["content"]

        # Wrap the result in a ChatResult with an AIMessage, which is what LangChain expects
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    # Property to indicate the type of LLM used in this wrapper
    @property
    def _llm_type(self) -> str:
        return "litellm"


# Function to return an instance of the custom LiteLLM wrapper
def get_litellm():
    return LiteLLMLangChainWrapper()
