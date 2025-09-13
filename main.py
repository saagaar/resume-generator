from typing import Union

from fastapi import FastAPI
from dotenv import load_dotenv
import os
import json
import getpass
from file_handler import extract_text
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

@app.get("/")
def convertToJson():
    resume_text = extract_text("cv/my_cv.pdf")
  # api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
  # model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
  # Following steps to workout tomoorow
  #1. Use ipynb to upload a file and save it to a destination
  #2. Extract the content from pdf or docx
  # 3. Convert it to json using LLM
  # 4. Save the json to a file
  # 5. use the content of json to rewrite the cover letter 
    schema = {
        "personal_information": {
        "name": "",
        "email": "",
        "phone": "",
        "location": ""
        },
        "summary": "",
        "skills": [],
        "experience": [
        {
            "role": "",
            "company": "",
            "start_date": "",
            "end_date": "",
            "description": []
        }
        ],
        "education": [
        {
            "degree": "",
            "institution": "",
            "year": ""
        }
        ]
    }

    # ⚙️ Initialize OpenRouter model (via OpenAI-compatible client)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # 📝 Prompt setup
    system = "You are a strict JSON extractor."
    human_prompt = """
    Extract the following CV into this JSON schema: {schema_content}

    Rules:
    - Copy content exactly into the schema.
    - Do not invent anything. Leave blank if missing.
    - Only return valid JSON.
    - Never use Markdown, code blocks, or ```json fencing.
    - Output raw JSON only.

    CV TEXT:
    {resume_text}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human_prompt)
    ])
    chain = prompt | llm

    # Run chain
    result = chain.invoke({"resume_text": resume_text,"schema_content": json.dumps(schema, indent=2)})
    content = result.content.strip()
    
    # Parse model output
    try:
            suggestions = json.loads(content)
            filename='cv/cv.json';
            with open(filename, "w") as file:
                json.dump(suggestions, file, indent=2)
            return (json.dumps(suggestions, indent=2)) 
    except json.JSONDecodeError:
        print(json.JSONDecodeError)
        return ("❌ Model returned invalid JSON:", json.JSONDecodeError)

@app.get("/suggestions")    
def suggestionsOnSkillSetJson():
  jd=(read_file('cv/jd.txt'))
  skillset= {
                "resume_improvements": "string - general advice on content, structure, tone, formatting",
                "skills_required": ["string", "string", ...],
                "experience_examples": ["string", "string", ...],
                "soft_skills": ["string", "string", ...],
                "certifications": ["string", "string", ...],
                "keywords_for_ATS": ["string", "string", ...]
              }
  system = """You are a strict JSON extractor.
              Always return output as **valid JSON only** following this schema: 
              {skillset}
            - Replace generic terms with industry-specific language.
            - Keep sentences short and natural with transitional phrases, like a human career coach speaking directly.       
            - Never include extra commentary outside the JSON.
            - Never use Markdown, code blocks, or ```json fencing.
            - Output raw JSON only.
            """
  human_prompt="""
                You are a professional career coach with decades of HR and recruitment experience. 
                I want to apply for the following position:
                {jd}
              Please provide me with up-to-date information on the skills and qualifications usually required for this role,
                examples of prior experience or work experience that would make me an exceptional candidate during a recruitment process for this role, 
              information on the soft skills, aptitudes and personality traits that employers are likely to recognize as valuable in this role, and suggestions for recognized certifications or training that would improve my chances of success 
              and 
              Suggest 5 keywords I should add to my resume to improve ATS compatibility in JSon format
  """
  llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash",
      temperature=0,
      max_tokens=None,
      timeout=None,
      max_retries=2,
  )
  finalprompt = ChatPromptTemplate.from_messages([
      ("system", system),
      ("human", human_prompt)
  ])
  chain = finalprompt | llm

  # Run chain
  result = chain.invoke({"jd": jd,"skillset":skillset})
  content = result.content.strip()
  try:
        suggestions = json.loads(content)
        filename='cv/suggestions.json';
        with open(filename, "w") as file:
            json.dump(suggestions, file, indent=2)
        return  (json.dumps(suggestions, indent=2)) 
  except json.JSONDecodeError:
      return ("❌ Model returned invalid JSON:")
      # print(result.content)   

@app.get("/application")    
def application(cv_file="cv/cv.json", suggestion_file="cv/suggestions.json", jd_file="cv/jd.txt"):
    context_file='cv/context_memory.json'
    conversation_history = []
    cv_data={}
    suggestion_data={}
    jd_text=""
    context_memory={}
    with open(cv_file, "r") as f:
        cv_data = json.load(f)
    with open(suggestion_file, "r") as f:
        suggestion_data = json.load(f)
    with open(jd_file, "r") as f:
        jd_text = f.read().strip()
    if os.path.exists("cv/context_memory.json"):
        with open(context_file, "r") as f:
            context_memory = f.read().strip()
    if context_memory=={}:
         with open(context_file, "w") as file   :
            context_memory={'cv_data':cv_data,'suggestions':suggestion_data,'job_description':jd_text,'qna':[]  }
            json.dump(context_memory, file, indent=2)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    # System prompt for the LLM
    system_prompt ="""You are an expert resume analyst and HR professional in tech.
    You have access to:
    - Candidate CV (skills, achievements)
    - Recruiter suggestions (role-specific advice, ATS keywords)
    - Job description
    Task:
    1. Ask the candidate clarifying questions one at a time to collect information for a personal statement.
    2. Only ask questions you need to generate a polished 150-word personal statement.
    3. Wait for candidate’s answer before asking the next question.
    4. Once you have enough info, output the final statement starting with "FINAL_STATEMENT:".
    5. Include relevant ATS keywords. Use a professional but human tone.
    """
    merged_context = {
            "cv_data": cv_data,
            "suggestions": suggestion_data,
            "job_description": jd_text,
            "qna": conversation_history
        }
    human_prompt = """
    Here is all available information about the candidate:
    {merged_context}
    Please generate the next clarifying question for the candidate, or output FINAL_STATEMENT: if you have enough information.
    """

    finalprompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    chain = finalprompt | llm
    result = chain.invoke({"merged_context": merged_context})
    with open(context_file, "r+") as f:
        context = f.read().strip()
    # return finalprompt.format_messages()
    
    # ✅ Use the structured prompt, not just the string
    conversation_history.append(result.content.strip())
    update_qna(question=result.content.strip()
)
    result.content.strip()
    # System prompt for the LLM
    system_prompt ="""You are an expert resume analyst and HR professional in tech.
    You have access to:
    - Candidate CV (skills, achievements)
    - Recruiter suggestions (role-specific advice, ATS keywords)
    - Job description
    Task:
    1. Ask the candidate clarifying questions one at a time to collect information for a personal statement.
    2. Only ask questions you need to generate a polished 150-word personal statement.
    3. Wait for candidate’s answer before asking the next question.
    4. Once you have enough info, output the final statement starting with "FINAL_STATEMENT:".
    5. Include relevant ATS keywords. Use a professional but human tone.
    """

def update_qna(question=None, answer=None):
    file_path = "cv/context_memory.json"
    # Load existing file or create new structure if not exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}   # keep other keys flexible
       
    # Ensure qna array exists
    if "qna" not in data:
        data["qna"] = []

    # Append new Q or A depending on what is passed
    if question and answer:
        # Add both together
        data["qna"].append({"question": question, "answer": answer})
    elif question:
        # Add a placeholder answer
        data["qna"].append({"question": question, "answer": ""})
    elif answer:
        # Attach answer to the last unanswered question
        if data["qna"] and data["qna"][-1]["answer"] == "":
            data["qna"][-1]["answer"] = answer
        else:
            # If no pending question, store as standalone
            data["qna"].append({"question": "", "answer": answer})

    # Save updated file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")




