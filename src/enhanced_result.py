from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class EnhancedResult(BaseModel):
    content: Union[str, Dict]
    metadata: Dict = Field(default_factory=dict)
    statistics: Dict = Field(default_factory=dict)
    screenshots: List[str] = Field(default_factory=list)
    ssl_certificate: Optional[Dict] = None
    downloaded_files: List[str] = Field(default_factory=list)
    media: Dict = Field(default_factory=dict)
    links: Dict = Field(default_factory=dict)
    markdown: Optional[str] = None
    extracted_content: Optional[Dict] = None