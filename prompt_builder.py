from langchain.prompts import PromptTemplate

def get_prompt():
    return PromptTemplate(
        input_variables=["resume", "job", "notes"],
        template="""
You're an expert resume reviewer. Given a resume and a job description, provide tailored improvement suggestions under these categories:

1. Summary
2. Skills
3. Experience
4. Formatting / ATS tips

--- RESUME ---
{resume}

--- JOB DESCRIPTION ---
{job}

--- USER NOTES ---
{notes}

Give specific, actionable feedback.
"""
    )
