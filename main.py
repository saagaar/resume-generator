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
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
def getLLM():
   return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
@app.get("/")
def convertCVToJson():
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
    
@app.get("/cv")
async def convertAndStreamCVToJson():
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

    async def generate():
        async for event in chain.astream({
            "resume_text": resume_text,
            "schema_content": json.dumps(schema, indent=2)
        }):
            if event.type == "token":
                token = event.content
                yield token  # 👈 stream chunk directly to client
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/suggestions")    
def getSuggestionsOnSkillSetJson():
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

@app.get("/bio")    
def getBio(cv_file="cv/cv.json", suggestion_file="cv/suggestions.json", jd_file="cv/jd.txt"):
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
    with open(context_file, "r") as f:
        full_context = f.read().strip()
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
    4. Make sure you have asked not more than 3 question that makes the best use of information for the given job description
    5. Once you have enough info, output the final statement starting with "FINAL_STATEMENT:".
    6. Include relevant ATS keywords. Use a professional but human tone.
    """
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
    result = chain.invoke({"merged_context": full_context})
    content= result.content.strip();
    if "FINAL_STATEMENT:" in content:
        final_statement = content.split("FINAL_STATEMENT:")[1].strip()
        with open(context_file, "r+") as f: 
            context_memory = f.read().strip()
        context_memory = json.loads(context_memory)
        context_memory['updated_cv']['bio'] = final_statement
        with open(context_file, "w") as f:
            json.dump(context_memory, f, indent=2)
        return final_statement

    else:   
        update_qna(question=result.content.strip())
    return content
    
    # return finalprompt.format_messages()
    
    # ✅ Use the structured prompt, not just the string
    # conversation_history.append(result.content.strip())
    
    return  result.content.strip()
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
@app.get("/answer") 
def submit_answer(answer: str, index=None):
     return update_qna(answer=answer, index=index);

@app.get("/question") 
def submit_question(question: str, index=None):
     update_qna(question=question, index=index);

def update_qna(question=None, answer=None, index=None):
    file_path = "cv/context_memory.json"

    # Load existing file or create new structure if not exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}   # keep other keys flexible
    if index is not None:
        index = int(index)
    # Ensure qna array exists
    if "qna" not in data:
        data["qna"] = []

    # Update by index if provided and valid
    if index is not None and 0 <= index < len(data["qna"]):
        if question:
            data["qna"][index]["question"] = question
        if answer:
            data["qna"][index]["answer"] = answer
    else:
        # Append new Q/A depending on what is passed
        if question and answer:
            data["qna"].append({"question": question, "answer": answer})
        elif question:
            data["qna"].append({"question": question, "answer": ""})
        elif answer:
            # Attach answer to the last unanswered question
            if data["qna"] and data["qna"][-1]["answer"] == "":
                data["qna"][-1]["answer"] = answer
            else:
                data["qna"].append({"question": "", "answer": answer})

    # Save updated file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

@app.get("/experience")
def getExperience():
    if os.path.exists("cv/context_memory.json"):
        with open("cv/context_memory.json", "r") as f:
            context_memory = f.read().strip()
            system_prompt ="""
                As an experienced recruiter in the tech HR space, for each job description in the cv_data from  with experience given, Please use this framework: 
                "Action Verb + Noun + Metric + [Strategy Optional] + Outcome = 1 bulleted achievement."
               
                - Write max three bullet point for each job description  that describes impact and uses metrics.
                - Make sure the job description is ats friendly and includes the suggestions in the key suggestions from the suggestions data. 
                - Never inclde generic terms make it specific to the job description. 
                - Never add random experience that is not in the cv_data
                - Use the experience information and make it relevant to job description but dont lie the skills and experience should be from the cv_data only. 
                - Use a professional tone but make it sound human. 
                      
            Now act as an experienced resume writer and career expert with a background in recruiting for [software engineer] roles in companies. 

            """
            human_prompt = """
            Here is all available information about the candidate:
            {context_memory}
            Please rewrite the job description for each of their past experience for the candidate
            - Limit the bullet point size to max 2 lines    
            - Never use Markdown, code blocks, or ```json fencing.
            - Json format for "experience": 
                    {{
                        "role": "",
                        "company": "",
                        "start_date": "",
                        "end_date": "",
                        "description": []
                    }}
            - Make sure the experience is relevant to the job description given in the context memory.  
            - Output only the  array above in plain json format.
            - If there is no experience given in the cv_data, return an empty array.
            - Output valid raw JSON only. 
            - Use keywords from the suggestions data to make it ats friendly.   
            """
            finalprompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            llm=getLLM()
            chain = finalprompt | llm
            result = chain.invoke({"context_memory": context_memory})
            content= result.content.strip();
            # Parse model output
            try:
                experience_json = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError("LLM did not return valid JSON:\n" + content)
            context_memory = json.loads(context_memory)
            context_memory['updated_cv']['experience'] = experience_json
            with open('cv/context_memory.json', "w") as f:
                json.dump(context_memory, f, indent=2)
            return experience_json
@app.get("/skills")
def skillset():
   if os.path.exists("cv/context_memory.json"):
        with open("cv/context_memory.json", "r") as f:
            context_memory = f.read().strip()
            system_prompt ="""
                You are an expert HR recruiter and career coach with 20+ years of experience in global industries, specializing in tech/software engineering.  
                Your task is to analyze a candidate’s CV data (structured JSON) and a given job description, then output the most recruiter-optimized skills grouped into categories.  

                ⚠️ RULES:
                - ONLY use top 3-5 skills per category that already exist in the CV data OR are explicitly mentioned in the `suggestions` field.  
                - Do NOT invent or hallucinate skills.
                - Only use top 5 most important visible skills that is
                - Dynamically create categories that make the MOST sense for recruiters and ATS in this specific role.
                - Within each category, sort skills by relevance and importance (prioritize overlap with the job description first, then strongest CV skills).
                - Ensure the output combines both **ATS keyword optimization** and **human recruiter readability** — meaning the skill list should maximize chances of passing automated scans *and* impressing a recruiter.
                - Output must be valid JSON only.  
                - No extra explanation outside JSON.
                - If there is no skills to match given, return an empty array.  
                - Never use Markdown, code blocks, or ```json fencing.

            """
            human_prompt = """
           Here is my CV, my suggestions, and my new resume draft, all in structured JSON.  
            Your task is to analyze these and produce a **recruiter-optimized and ATS-friendly skill list**.  

            ⚠️ Instructions:
            1. Only use skills present in the CV, suggestions, or new resume draft for the main skill list (`recommended_skills`).  
            2. Do not hallucinate or invent new skills in the main skill list.  
            3. Group skills into **categories that make sense for recruiters**, dynamically decided based on the role and industry.  
            4. Order both categories and skills so that **the most relevant to the job description come first**.  
            5. Make skill names **resume-friendly**, ATS-compatible, and human-readable.  
            6. Additionally, create a `suggested_skills` array: optional skills **not in my current CV** but recommended for this job to boost my resume.  
            7. Output must be **valid JSON only**, no extra text or commentary.  
            8. Make sure the skills are relevant to the position applied for; e.g. if it's senior position skills like git or agile may be too general or obvious you can avoid them; instead of generic like agile use scrum or kanban

            ---

            [Full JSON Input]:
            {context_memory}    
            """
            finalprompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            llm=getLLM()
            chain = finalprompt | llm
            result = chain.invoke({"context_memory": context_memory})
            content= result.content.strip();
            # Parse model output
            try:
                skillsList = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError("LLM did not return valid JSON:\n" + content)
            context_memory = json.loads(context_memory)
            context_memory['updated_cv']['skills'] = skillsList
            with open('cv/context_memory.json', "w") as f:
                json.dump(context_memory, f, indent=2)
            return skillsList
@app.get("/review")
def resumeReviewer():
    if os.path.exists("cv/context_memory.json"):
        with open("cv/context_memory.json", "r") as f:
            context_memory = f.read().strip()
            system_prompt ="""
                You are a highly experienced senior career coach and resume analyst. Your task is to critically analyze CVs for software engineering roles, focusing on technical expertise, leadership, and alignment with job requirements. You will act as a professional reviewer who provides actionable, structured, and relevant suggestions for improvement, acknowledges strengths, and highlights areas to optimize for ATS and human readers.  
                Always follow these rules:  
                1. Treat the CV JSON as the authoritative source of information; do not hallucinate experience or skills.  
                2. Consider both technical and soft skills, certifications, and measurable achievements.  
                3. Compare the CV with the provided job description and identify any gaps, improvements, or strengths.  
                4. Focus on clarity, structure, readability, and alignment with the employer's expectations.  
                5. Output results in JSON format with fields for overall feedback, critical suggestions, and acknowledgements.  
                6. Provide additional suggested skills if they are relevant to the job description and can strengthen the CV.  
            """
            human_prompt = """
           Review the following CV JSON in the context of the target job description. Act as a senior resume analyst with decades of experience. Critically evaluate the CV, highlight strengths, identify actionable improvements, and suggest skills that can enhance the CV’s alignment with the role.  
            CV ,Suggestions and job description are given in the context memory:
           
            {context_memory}
            * Remember that the new CV is in context memory as `updated_cv`. Make sure you review this not the cv_data

            Instructions:  
            1. Analyze technical skills, frameworks, programming languages, cloud, DevOps, architecture, Agile, and TDD experience.  
            2. Review experience bullet points for quantifiable achievements and alignment with the job description.  
            3. Identify gaps in skills or experience relative to the job description.  
            4. Suggest additional skills that would boost the CV for this role without inventing experience.  
            5. Acknowledge strengths and well-presented sections of the CV.  
            6. Provide feedback as structured JSON in this format:
            7. Never use Markdown, code blocks, or ```json fencing.
            8.Only mention important details a short line or two for each point

            ```json
            {{
            "overall_feedback": "Summary of how well the CV fits the job description",
            "critical_suggestions": [
                "List of actionable improvements, if any"
            ],
            "acknowledgements": [
                "List of strengths and well-presented aspects"
            ],
            "suggested_skills_to_add": [
                "Optional: list of relevant skills to boost CV alignment"
            ]
            }}
            
            """
            finalprompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            llm=getLLM()
            chain = finalprompt | llm
            result = chain.invoke({"context_memory": context_memory})
            content= result.content.strip();
            # Parse model output
            try:
                finalSuggestions = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError("LLM did not return valid JSON:\n" + content)
            context_memory = json.loads(context_memory)
            context_memory['updated_cv']['finalSuggestions'] = finalSuggestions
            with open('cv/context_memory.json', "w") as f:
                json.dump(context_memory, f, indent=2)
            return finalSuggestions    
@app.get('/coverletter')
def coverLetter():
     if os.path.exists("cv/context_memory.json"):
        with open("cv/context_memory.json", "r") as f:
            context_memory = f.read().strip()
            system_prompt ="""
             You are a highly experienced senior tech recruiter and career consultant. Your task is to write a professional, persuasive, and highly effective cover letter. Follow these rules:
                1. Research the target company and job position provided.  
                2. Understand the company’s mission, values, and objectives to tailor the cover letter.  
                3. Identify the most relevant contact person for the position (recruiter, hiring manager, LinkedIn profile, or email).  
                4. Use the skills listed in the provided JSON suggestions for ATS to highlight qualifications.  
                5. Use the CV information provided, emphasizing measurable achievements, leadership, and technical expertise.  
                6. Write a concise, personalized, and compelling cover letter with a clear introduction, body, and closing.  
                7. Avoid generic phrasing; align content with the company’s mission and the role’s responsibilities.  
                8. Output ONLY in JSON format using this schema:

               
                {{
                "cover_letter_text": "Full cover letter text ready to send",
                "contact_person": "Name, role, LinkedIn URL/email if available",
                "highlighted_skills": ["List of top skills used in cover letter"],
                "company_objective_summary": "Brief summary of company objectives or mission incorporated into cover letter"
                }}
            """
            human_prompt = """
            Write a professional cover letter based on the following information:```text
            Use the following information to generate a personalized, high-impact cover letter. Follow all instructions from the system prompt.

            CV JSON:
            
            {context_memory}
           
            Instructions:
            1. Extract ATS skills from the JSON suggestions and incorporate them naturally.  
            2. Highlight achievements, leadership, and technical expertise relevant to the job.  
            3. Incorporate the company’s goals, mission, and values in the cover letter.  
            4. Suggest the most appropriate contact person for addressing the letter.  
            5. Output ONLY in JSON format using the schema from the system prompt.
            6. Never use Markdown, code blocks, or ```json fencing.                
            """
            finalprompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            llm=getLLM()
            chain = finalprompt | llm
            result = chain.invoke({"context_memory": context_memory})
            content= result.content.strip();
            # Parse model output
            try:
                cover = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError("LLM did not return valid JSON:\n" + content)
            context_memory = json.loads(context_memory)
            context_memory['updated_cv']['coverLetter'] = cover
            with open('cv/context_memory.json', "w") as f:
                json.dump(context_memory, f, indent=2)
            return cover    
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")




