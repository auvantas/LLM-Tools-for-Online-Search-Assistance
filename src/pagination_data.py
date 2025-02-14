from pydantic import BaseModel, Field
from typing import List, Optional

class PaginationData(BaseModel):
    page_urls: List[str] = Field(default_factory=list)
    next_page: Optional[str] = None
    page_pattern: Optional[str] = None