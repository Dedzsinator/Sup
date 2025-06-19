// Types and interfaces for the Sup messaging application

export interface User {
  id: string;
  email: string;
  username: string;
  avatar_url?: string;
  is_online: boolean;
  last_seen?: string;
}

export interface Room {
  id: string;
  name: string;
  description?: string;
  type: 'group' | 'direct_message' | 'channel';
  is_private: boolean;
  avatar_url?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  unread_count?: number;
}

export interface Message {
  id: string;
  sender_id: string;
  room_id: string;
  content: string;
  type: 'text' | 'image' | 'file' | 'audio' | 'video';
  timestamp: string;
  inserted_at: string;
  reply_to_id?: string;
  edited_at?: string;
  delivery_status?: 'sent' | 'delivered' | 'read';
  sender?: User;
}

export interface DeliveryReceipt {
  id: string;
  message_id: string;
  user_id: string;
  status: 'sent' | 'delivered' | 'read';
  sent_at?: string;
  delivered_at?: string;
  read_at?: string;
}

export interface TypingIndicator {
  user_id: string;
  room_id: string;
  is_typing: boolean;
  timestamp: string;
}

export interface PresenceUpdate {
  user_id: string;
  status: 'online' | 'offline';
  timestamp: string;
}

export interface WebSocketMessage {
  type: 'message' | 'typing' | 'presence' | 'delivery_receipt' | 'error' | 'message_sent';
  data: any;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface ChatState {
  rooms: Room[];
  messages: { [roomId: string]: Message[] };
  currentRoom: Room | null;
  typingUsers: { [roomId: string]: string[] };
  onlineUsers: { [userId: string]: boolean };
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  success: boolean;
}

export interface SendMessageRequest {
  room_id: string;
  content: string;
  type: 'text' | 'image' | 'file' | 'audio' | 'video';
  reply_to_id?: string;
}

export interface CreateRoomRequest {
  name: string;
  type: 'group' | 'channel';
  description?: string;
  is_private?: boolean;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}
