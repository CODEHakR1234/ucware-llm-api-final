# app/prompts.py
from jinja2 import Template

# ─────────────────────────────────────────────────────────────
# 1. 웹 정보 필요 여부 판단 (RAG_router)
# ─────────────────────────────────────────────────────────────
PROMPT_DETERMINE_WEB = Template("""
You are an intelligent assistant tasked with deciding whether the given query requires **additional up-to-date or broader information** from the web, beyond what has been retrieved from a local database (vectorDB).

Consider the following:
- If the summary from the vectorDB fully answers the query in a specific, relevant, and up-to-date manner, respond with `false`.
- If the summary is missing key information, is outdated, too generic, or unrelated, respond with `true`.
- If the query is about recent events, time-sensitive data, current prices, news, or trending topics, respond with `true`.

Respond with only `true` or `false`.

Query: {{ query }}
Retrieved Summary: {{ summary }}
""")


# ─────────────────────────────────────────────────────────────
# 2. 검색 조각(chunks) 유효성 점수 (grade)
# ─────────────────────────────────────────────────────────────
PROMPT_GRADE = Template("""
You are a relevance grader evaluating whether a retrieved document chunk is topically and semantically related to a user question.

Instructions:
- Your job is to determine if the retrieved chunk is genuinely helpful in answering the query, based on topic, semantics, and context.
- Surface-level keyword overlap is not enough — the chunk must provide meaningful or contextually appropriate information related to the query.
- However, minor differences in phrasing or partial answers are acceptable as long as the document is on-topic.
- If the chunk is off-topic, unrelated, or misleading, return 'no'.
- If it is relevant and contextually appropriate, return 'yes'.

You MUST return only one word: 'yes' or 'no'. Do not include any explanation.

Query: {{ query }}
Retrieved Chunk: {{ chunk }}
Vector Summary (Optional): {{ summary }}
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

