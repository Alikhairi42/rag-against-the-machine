import uuid
from typing import List, Union
from pydantic import BaseModel, Field

# 1. MinimalSource Model
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

# 👉 DABA KMEEL NTA:
# Zid: UnansweredQuestion
# Zid: AnsweredQuestion
# Zid: RagDataset
# Zid: MinimalSearchResults
# Zid: MinimalAnswer
# Zid: StudentSearchResults
# Zid: StudentSearchResultsAndAnswer