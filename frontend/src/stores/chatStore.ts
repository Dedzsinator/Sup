import { create } from 'zustand';
import { Room, Message, ChatState, SendMessageRequest } from '../types';
import { apiClient } from '../services/api';
import { webSocketService } from '../services/websocket';

interface ChatStore extends ChatState {
  // Room management
  loadRooms: () => Promise<void>;
  createRoom: (name: string, type: 'group' | 'channel', description?: string) => Promise<boolean>;
  joinRoom: (roomId: string) => Promise<boolean>;
  leaveRoom: (roomId: string) => Promise<boolean>;
  setCurrentRoom: (room: Room | null) => void;
  
  // Message management
  loadMessages: (roomId: string, limit?: number, before?: string) => Promise<void>;
  sendMessage: (roomId: string, content: string, type?: 'text' | 'image' | 'file', replyToId?: string) => void;
  addMessage: (message: Message) => void;
  updateMessageStatus: (messageId: string, status: 'sent' | 'delivered' | 'read') => void;
  
  // Real-time features
  startTyping: (roomId: string) => void;
  stopTyping: (roomId: string) => void;
  setTypingUsers: (roomId: string, userIds: string[]) => void;
  setUserOnline: (userId: string, isOnline: boolean) => void;
  
  // Search
  searchMessages: (query: string) => Promise<Message[]>;
  
  // Initialization
  initialize: () => void;
  cleanup: () => void;
}

export const useChatStore = create<ChatStore>((set, get) => ({
  rooms: [],
  messages: {},
  currentRoom: null,
  typingUsers: {},
  onlineUsers: {},

  initialize: () => {
    // Set up WebSocket event listeners
    webSocketService.on('message', (message: Message) => {
      get().addMessage(message);
    });

    webSocketService.on('typing', (typing: any) => {
      const currentTyping = get().typingUsers[typing.room_id] || [];
      if (typing.is_typing) {
        if (!currentTyping.includes(typing.user_id)) {
          get().setTypingUsers(typing.room_id, [...currentTyping, typing.user_id]);
        }
      } else {
        get().setTypingUsers(typing.room_id, currentTyping.filter(id => id !== typing.user_id));
      }
    });

    webSocketService.on('presence', (presence: any) => {
      get().setUserOnline(presence.user_id, presence.status === 'online');
    });

    webSocketService.on('delivery_receipt', (receipt: any) => {
      get().updateMessageStatus(receipt.message_id, receipt.status);
    });

    webSocketService.on('message_sent', (message: Message) => {
      // Update local message with server confirmation
      const state = get();
      const roomMessages = state.messages[message.room_id] || [];
      const updatedMessages = roomMessages.map(msg => 
        msg.id === message.id ? { ...msg, delivery_status: 'sent' } : msg
      );
      
      set({
        messages: {
          ...state.messages,
          [message.room_id]: updatedMessages,
        },
      });
    });
  },

  cleanup: () => {
    // Remove WebSocket listeners
    webSocketService.off('message', () => {});
    webSocketService.off('typing', () => {});
    webSocketService.off('presence', () => {});
    webSocketService.off('delivery_receipt', () => {});
    webSocketService.off('message_sent', () => {});
  },

  loadRooms: async () => {
    try {
      const response = await apiClient.getRooms();
      if (response.success && response.data) {
        set({ rooms: response.data.rooms });
      }
    } catch (error) {
      console.error('Failed to load rooms:', error);
    }
  },

  createRoom: async (name: string, type: 'group' | 'channel', description?: string): Promise<boolean> => {
    try {
      const response = await apiClient.createRoom({ name, type, description });
      if (response.success && response.data) {
        const state = get();
        set({ rooms: [...state.rooms, response.data] });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to create room:', error);
      return false;
    }
  },

  joinRoom: async (roomId: string): Promise<boolean> => {
    try {
      const response = await apiClient.joinRoom(roomId);
      if (response.success) {
        webSocketService.joinRoom(roomId);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to join room:', error);
      return false;
    }
  },

  leaveRoom: async (roomId: string): Promise<boolean> => {
    try {
      const response = await apiClient.leaveRoom(roomId);
      if (response.success) {
        const state = get();
        set({
          rooms: state.rooms.filter(room => room.id !== roomId),
          currentRoom: state.currentRoom?.id === roomId ? null : state.currentRoom,
        });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to leave room:', error);
      return false;
    }
  },

  setCurrentRoom: (room: Room | null) => {
    set({ currentRoom: room });
    if (room) {
      // Load messages for the room if not already loaded
      const state = get();
      if (!state.messages[room.id]) {
        get().loadMessages(room.id);
      }
    }
  },

  loadMessages: async (roomId: string, limit = 50, before?: string) => {
    try {
      const response = await apiClient.getRoomMessages(roomId, limit, before);
      if (response.success && response.data) {
        const state = get();
        const existingMessages = state.messages[roomId] || [];
        const newMessages = before 
          ? [...response.data.messages, ...existingMessages]
          : response.data.messages;
        
        set({
          messages: {
            ...state.messages,
            [roomId]: newMessages,
          },
        });
      }
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  },

  sendMessage: (roomId: string, content: string, type = 'text', replyToId?: string) => {
    // Create optimistic message
    const optimisticMessage: Message = {
      id: `temp-${Date.now()}`,
      sender_id: '', // Will be set by auth store
      room_id: roomId,
      content,
      type,
      timestamp: new Date().toISOString(),
      reply_to_id: replyToId,
      delivery_status: 'sent',
    };

    // Add to local state immediately
    get().addMessage(optimisticMessage);

    // Send via WebSocket
    webSocketService.sendMessage(roomId, content, type, replyToId);
  },

  addMessage: (message: Message) => {
    const state = get();
    const roomMessages = state.messages[message.room_id] || [];
    
    // Check if message already exists (avoid duplicates)
    const existingIndex = roomMessages.findIndex(msg => msg.id === message.id);
    
    let updatedMessages;
    if (existingIndex >= 0) {
      // Replace existing message
      updatedMessages = [...roomMessages];
      updatedMessages[existingIndex] = message;
    } else {
      // Add new message (maintain chronological order)
      updatedMessages = [...roomMessages, message].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
    }

    set({
      messages: {
        ...state.messages,
        [message.room_id]: updatedMessages,
      },
    });
  },

  updateMessageStatus: (messageId: string, status: 'sent' | 'delivered' | 'read') => {
    const state = get();
    const updatedMessages = { ...state.messages };
    
    Object.keys(updatedMessages).forEach(roomId => {
      updatedMessages[roomId] = updatedMessages[roomId].map(msg =>
        msg.id === messageId ? { ...msg, delivery_status: status } : msg
      );
    });

    set({ messages: updatedMessages });
  },

  startTyping: (roomId: string) => {
    webSocketService.startTyping(roomId);
  },

  stopTyping: (roomId: string) => {
    webSocketService.stopTyping(roomId);
  },

  setTypingUsers: (roomId: string, userIds: string[]) => {
    const state = get();
    set({
      typingUsers: {
        ...state.typingUsers,
        [roomId]: userIds,
      },
    });
  },

  setUserOnline: (userId: string, isOnline: boolean) => {
    const state = get();
    set({
      onlineUsers: {
        ...state.onlineUsers,
        [userId]: isOnline,
      },
    });
  },

  searchMessages: async (query: string): Promise<Message[]> => {
    try {
      const response = await apiClient.searchMessages(query);
      if (response.success && response.data) {
        return response.data.messages;
      }
      return [];
    } catch (error) {
      console.error('Failed to search messages:', error);
      return [];
    }
  },
}));
