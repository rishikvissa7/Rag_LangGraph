from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import litellm

class LiteLLMLangChainWrapper(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Accept Human, System, and AI messages
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                prompt_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"AI: {msg.content}")
            elif isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        prompt = "\n".join(prompt_parts)

        # Call LiteLLM API
        response = litellm.completion(
            model="together_ai/mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role": "user", "content": prompt}],
            api_key="309ce89cad6d001f10acf44e5e5bed429ab3cea28b0a5167de6e961e7c97f7eb"
        )

        content = response["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "litellm"



def get_litellm():
    return LiteLLMLangChainWrapper()
