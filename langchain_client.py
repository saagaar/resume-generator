from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompt_builder import get_prompt
import os
from dotenv import load_dotenv

load_dotenv()

def generate_suggestions(resume_text, jd_text, user_notes=""):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5) # type: ignore
    prompt = get_prompt()
    print(prompt);
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({
        "resume": resume_text,
        "job": jd_text,
        "notes": user_notes
    })

    return result

def generate_suggestions_deepseek(resume_text, jd_text, user_notes=""):
    # Load DeepSeek-specific configs
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Create LLM instance using DeepSeek
    llm = ChatOpenAI(
        model=deepseek_model,
        temperature=0.5,
        openai_api_key=deepseek_api_key,
        openai_api_base=deepseek_api_base
    ) # type: ignore

    # Get prompt
    prompt = get_prompt()
    if prompt is None:
        raise ValueError("Prompt from get_prompt() is None. Please check your prompt_builder module.")

    # Create and run LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({
        "resume": resume_text,
        "job": jd_text,
        "notes": user_notes
    })

    return result

def generate_suggestions_openrouter(resume_text, jd_text, user_notes=""):
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in environment")

    llm = ChatOpenAI(
        model=model,
        temperature=0.5,
        openai_api_key=api_key,
        openai_api_base=api_base
    )#type: ignore

    prompt = get_prompt()
    if prompt is None:
        raise ValueError("Prompt is None. Check get_prompt() implementation.")

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({
        "resume": resume_text,
        "job": jd_text,
        "notes": user_notes
    })

    return result