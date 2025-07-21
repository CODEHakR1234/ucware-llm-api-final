from pydantic import BaseModel, HttpUrl

class SummaryRequestDTO(BaseModel):
    file_id: str
    pdf_url: HttpUrl
    query: str
    lang: str
