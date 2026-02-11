from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class StreamingTokenBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.current_tokens = 0
    
    def check_and_increment(self, chunk_tokens: int):
        self.current_tokens += chunk_tokens
        if self.current_tokens > self.max_tokens:
            raise TokenBudgetExceededError(
                f"Token budget exceeded mid-generation: {self.current_tokens}/{self.max_tokens}"
            )

class TokenBudgetExceededError(Exception):
    pass

def stream_with_budget(chain, input_data, budget: StreamingTokenBudget):
    """Stream response and halt if budget exceeded"""
    result = ""
    try:
        for chunk in chain.stream(input_data):
            chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_tokens = len(chunk_text.split()) * 1.3  # Rough estimate
            
            budget.check_and_increment(int(chunk_tokens))
            result += chunk_text
            print(chunk_text, end="", flush=True)
        
        return result
    except TokenBudgetExceededError as e:
        print(f"\n\n❌ HALTED: {e}")
        return result

# Usage
llm = ChatOpenAI(streaming=True)
prompt = ChatPromptTemplate.from_template("Write a long essay about: {topic}")
chain = prompt | llm

budget = StreamingTokenBudget(max_tokens=50)  # Low limit for testing

try:
    result = stream_with_budget(
        chain, 
        {"topic": "artificial intelligence"}, 
        budget
    )
except TokenBudgetExceededError:
    pass