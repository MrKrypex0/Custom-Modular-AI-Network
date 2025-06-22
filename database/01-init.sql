-- Create database if it doesn't exist
CREATE DATABASE gpt2_chat;

-- Connect to the database
\c gpt2_chat;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    avatar_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Create chat_rooms table
CREATE TABLE IF NOT EXISTS chat_rooms (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    type VARCHAR(50) DEFAULT 'text',
    user_id INTEGER REFERENCES users(id),
    chat_room_id INTEGER REFERENCES chat_rooms(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performence
CREATE INDEX IF NOT EXISTS idx_messages_chat_room_id ON messages(chat_room_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- INSERT sample data
INSERT INTO users (name. email) VALUES
    ('John Doe', 'john@example.com'),
    ('Jane Smith', 'jane@example.com');
    ('GPT-2 Bot', 'bot@gpt2.com')
ON CONFLICT (email) DO NOTHING;

INSERT INTO chat_rooms (name, description, created_by) VALUES 
    ('General', 'General discussion room', 1),
    ('Tech Talk', 'Technology discussions', 2),
    ('Random', 'Random conversations', 1)
ON CONFLICT DO NOTHING

INSERT INTO messages (content, type, user_id, chat_room_id) VALUES 
    ('Welcome to the chat!', 'text', 1, 1),
    ('Hello everyone!', 'text', 2, 1),
    ('How is everyone doing?', 'text', 3, 1)
ON CONFLICT DO NOTHING;