// Types and interfaces for the Sup messaging application

export interface User {
  id: string;
  email: string;
  username: string;
  display_name?: string;
  avatar_url?: string;
  profile_banner_url?: string;
  bio?: string;
  status_message?: string;
  is_online: boolean;
  last_seen?: string;
  activity_status: 'online' | 'away' | 'busy' | 'invisible';
  custom_activity?: {
    type: 'playing' | 'listening' | 'watching' | 'custom';
    name: string;
    details?: string;
  };
  theme_preference: 'system' | 'light' | 'dark';
  accent_color: string;
  date_joined: string;
  friend_code: string;
  email_verified: boolean;
  phone_verified: boolean;
  two_factor_enabled: boolean;
}

export interface UserSettings {
  notification_settings: {
    messages: boolean;
    mentions: boolean;
    calls: boolean;
    sound: boolean;
    vibration: boolean;
    email_notifications: boolean;
  };
  privacy_settings: {
    online_status: 'everyone' | 'friends' | 'nobody';
    profile_visibility: 'everyone' | 'friends' | 'nobody';
    message_receipts: boolean;
    typing_indicators: boolean;
  };
  call_settings: {
    camera_default: boolean;
    mic_default: boolean;
    noise_suppression: boolean;
    echo_cancellation: boolean;
    video_quality: 'auto' | 'low' | 'medium' | 'high';
  };
}

export interface Friendship {
  id: string;
  requester: User;
  addressee?: User;
  status: 'pending' | 'accepted' | 'blocked';
  created_at: string;
}

export interface FriendRequest {
  id: string;
  requester: User;
  created_at: string;
}

export interface Call {
  id: string;
  caller_id: string;
  room_id?: string;
  type: 'voice' | 'video' | 'screen_share';
  status: 'connecting' | 'ringing' | 'active' | 'ended' | 'missed' | 'declined';
  started_at: string;
  ended_at?: string;
  duration?: number;
  participants: string[];
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
  type: 'message' | 'typing' | 'presence' | 'delivery_receipt' | 'error' | 'message_sent' | 'call_initiated' | 'call_answered' | 'call_declined' | 'call_ended' | 'webrtc_signaling';
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

export interface CallState {
  currentCall: Call | null;
  incomingCall: Call | null;
  callHistory: Call[];
  isCallActive: boolean;
  localStream: MediaStream | null;
  remoteStreams: { [userId: string]: MediaStream };
  audioEnabled: boolean;
  videoEnabled: boolean;
  screenShareEnabled: boolean;
}

export interface FriendsState {
  friends: User[];
  friendRequests: FriendRequest[];
  sentRequests: FriendRequest[];
  blockedUsers: User[];
  isLoading: boolean;
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
  display_name?: string;
  avatar_url?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface UpdateProfileRequest {
  display_name?: string;
  bio?: string;
  status_message?: string;
  avatar_url?: string;
  profile_banner_url?: string;
  theme_preference?: 'system' | 'light' | 'dark';
  accent_color?: string;
  activity_status?: 'online' | 'away' | 'busy' | 'invisible';
  custom_activity?: {
    type: 'playing' | 'listening' | 'watching' | 'custom';
    name: string;
    details?: string;
  };
}

export interface UpdateSettingsRequest {
  notification_settings?: Partial<UserSettings['notification_settings']>;
  privacy_settings?: Partial<UserSettings['privacy_settings']>;
  call_settings?: Partial<UserSettings['call_settings']>;
}

export interface InitiateCallRequest {
  room_id?: string;
  type: 'voice' | 'video';
  participants: string[];
}

export interface WebRTCSignaling {
  type: 'offer' | 'answer' | 'ice_candidate';
  from_user_id: string;
  data: any;
}
