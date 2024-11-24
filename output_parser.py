from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Anlogy(BaseModel):
    word : str = Field(description="문맥에 맞는 단어의 뜻을 출력해주세요")
    analysis : str = Field(description="전체 문장의 해석을 출력해주세요")
    def to_dict(self):
        return {"word": self.word, "analysis": self.analysis}

analogy = PydanticOutputParser(pydantic_object=Anlogy)