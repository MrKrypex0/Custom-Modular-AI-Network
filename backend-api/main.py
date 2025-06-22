from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
from dotenv import load_dotenv

from database import SessionLocal, engine
from models import Base, User, ChatRoom, Message
from schemas import UserCreate, UserResponse, ChatRoomResponse, MessageCreate, MessageResponse

load_dotenv()

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="GPT-2 Chat API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "GPT-2 Chat API"}

@app.get("/api/users", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.post("/api/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/api/chatrooms", response_model=List[ChatRoomResponse])
async def get_chatrooms(db: Session = Depends(get_db)):
    chatrooms = db.query(ChatRoom).all()
    return chatrooms

@app.get("/api/messages", response_model=List[MessageResponse])
async def get_messages(db: Session = Depends(get_db)):
    messages = db.query(Message).all()
    return messages

@app.post("/api/messages", response_model=MessageResponse)
async def create_message(message: MessageCreate, db: Session = Depends(get_db)):
    db_message = Message(**message.dict())
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
