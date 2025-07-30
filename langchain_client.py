from langchain_openai import ChatOpenAI
from prompt_builder import get_prompt,get_prompt_old
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from file_handler import extract_text


import os
from dotenv import load_dotenv

load_dotenv()

def generate_suggestions(resume_text, jd_text, user_notes=""):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5) # type: ignore
    prompt = get_prompt_old()
   
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
    prompt = get_prompt_old()
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

def generate_suggestions_openrouter(resume_text='', jd_text='', user_notes=""):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment")
        print(api_key);
        llm = ChatOpenAI(
            model="meta-llama/llama-3-8b-instruct",  # Or any other available model
            temperature=0.5,
            openai_api_key= api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        ) # type: ignore

        prompt = get_prompt()
        
        print('*****Prompt','*****Promptend')
        filled_prompt = prompt.format(
            resume=resume_text,
            job=jd_text,    
            notes=user_notes
        )  
        print('%%%%',filled_prompt,'%%%%');
        if prompt is None:
            raise ValueError("Prompt is None. Check get_prompt() implementation.")
    
        response = llm.invoke(filled_prompt)
       
        if hasattr(response, "content"):
            return response.content
        return response; #result.to_messages()[0].content if result else None
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return None
    
def test():
    print('k xa')

if __name__ == "__main__":
    import sys
    dummy_resume = "Experienced backend developer with expertise in Python and cloud infrastructure."
    dummy_jd = "We're hiring a Python developer familiar with AWS and scalable systems."
    dummy_notes = "Emphasize experience with REST APIs and CI/CD pipelines."
    result = generate_suggestions_openrouter(dummy_resume, dummy_jd, dummy_notes)
    print(result)
    test()  