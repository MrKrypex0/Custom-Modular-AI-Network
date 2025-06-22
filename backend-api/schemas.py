from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    name: str
    email: EmailStr
    avatar_url: Optional[str] = None

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatRoomBase(BaseModel):
    name: str
    description: Optional[str] = None

class ChatRoomResponse(ChatRoomBase):
    id: int
    created_by: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MessageBase(BaseModel):
    content: str
    type: str = "text"
    user_id: Optional[int] = 1  # Default to user 1 for now
    chat_room_id: Optional[int] = 1  # Default to room 1 for now

class MessageCreate(MessageBase):
    pass

class MessageResponse(MessageBase):
    id: int
    created_at: datetime
    updated_at: datetime
    user: Optional[UserResponse] = None
    isCurrentUser: bool = True  # For frontend compatibility

    class Config:
        from_attributes = True
