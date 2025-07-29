from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# models/client.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
# from db.base import Base
from datetime import datetime

class Client(Base):
    __tablename__ = "client"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    ip_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="client")


# models/session.py
from sqlalchemy import Column, Integer, DateTime, Enum as SqlEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

class ConversationStatusEnum(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"

from typing import List, Literal
from pydantic import BaseModel

class Message(BaseModel):
    sender: Literal["user", "bot"]
    content: str

class UserMessage(BaseModel):
    session_id: str
    messages: List[Message]
    collection_name: str = "client"
    image_url: str | None = None
