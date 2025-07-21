# app/prompts.py
from jinja2 import Template

# ─────────────────────────────────────────────────────────────
# 1. 웹 정보 필요 여부 판단 (RAG_router)
# ─────────────────────────────────────────────────────────────
PROMPT_DETERMINE_WEB = Template("""
You are a helpful assistant that can determine if the answer of the query need extra information from the web.
If the answer need extra information from the web, return 'true'.
If the answer does not need extra information from the web, return 'false'.
Query: {{ query }}
Summary: {{ summary }}
""")

# ─────────────────────────────────────────────────────────────
# 2. 검색 조각(chunks) 유효성 점수 (grade)
# ─────────────────────────────────────────────────────────────
PROMPT_GRADE = Template("""
You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
YOU MUST RETURN ONLY 'yes' or 'no'.
Query: {{ query }}
Summary: {{ summary }}
Retrieved: {{ chunk }}
""")

# ─────────────────────────────────────────────────────────────
# 3. 최종 답변 생성 (generate)
# ─────────────────────────────────────────────────────────────
PROMPT_GENERATE = Template("""
You are a helpful assistant that can generate a answer of the query in English.
Use the retrieved information to generate the answer.
YOU MUST RETURN ONLY THE ANSWER, NOTHING ELSE.
Query: {{ query }}
Retrieved: {{ retrieved }}
""")

# ─────────────────────────────────────────────────────────────
# 4. 답변 품질 검증 (verify)
# ─────────────────────────────────────────────────────────────
PROMPT_VERIFY = Template("""
You are a helpful assistant that can verify the quality of the generated answer.
Please evaluate the answer based on the following criteria:
1. Does the answer directly address the query?
2. Is the answer based on the retrieved information?
3. Is the answer logically consistent?
4. Is the answer complete and specific?

Query: {{ query }}
Summary: {{ summary }}
Retrieved Information: {{ retrieved }}
Generated Answer: {{ answer }}

Return 'good' if the answer meets all criteria, otherwise return 'bad'. Do not return anything else.
""")

# ─────────────────────────────────────────────────────────────
# 5. 쿼리 리파인 또는 사과문 (refine)
# ─────────────────────────────────────────────────────────────
PROMPT_REFINE = Template("""
You are a helpful assistant that can do two things:
1. If the query is not related to the document content, return ONLY this sentence: "I'm sorry, I can't find the answer to your question even though I read all the documents. Please ask a question about the document's content."
2. If the query is related, refine the query to get more relevant and accurate information based on the document summary and retrieved information. Return ONLY the refined query, nothing else.

Document Summary: {{ summary }}
Original Query: {{ query }}
Retrieved Information: {{ retrieved }}
Generated Answer: {{ answer }}
""")

# ─────────────────────────────────────────────────────────────
# 6. 번역 (translate)
# ─────────────────────────────────────────────────────────────
PROMPT_TRANSLATE = Template("""
You are a helpful assistant that can translate the answer to User language.
ONLY RETURN THE TRANSLATED SEQUENCE, NOTHING ELSE.
User language: {{ lang }}
Answer: {{ text }}
""")

