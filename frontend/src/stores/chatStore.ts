import { create } from 'zustand';
import { 
  Room, 
  Message, 
  ChatState, 
  MessageReaction, 
  MessageThread, 
  CustomEmoji, 
  OfflineMessage 
} from '../types';
import { apiClient } from '../services/api';
import { webSocketService } from '../services/websocket';

interface ExtendedChatState extends ChatState {
  // Thread and reaction state
  messageReactions: { [messageId: string]: MessageReaction[] };
  messageThreads: { [messageId: string]: MessageThread };
  threadMessages: { [threadId: string]: Message[] };
  
  // Custom emoji state
  roomEmojis: { [roomId: string]: CustomEmoji[] };
  globalEmojis: CustomEmoji[];
  
  // Offline message state
  offlineMessages: OfflineMessage[];
  
  // Search state
  searchResults: Message[];
  searchQuery: string;
  
  // Spam detection state
  spamDetectionEnabled: boolean;
  spamStats: any;
  isSpamServiceHealthy: boolean;
  
  // UI state
  selectedThread: MessageThread | null;
  editingMessage: Message | null;
  replyingToMessage: Message | null;
}

interface ChatStore extends ExtendedChatState {
  // Room management
  loadRooms: () => Promise<void>;
  createRoom: (name: string, type: 'group' | 'channel', description?: string) => Promise<boolean>;
  joinRoom: (roomId: string) => Promise<boolean>;
  leaveRoom: (roomId: string) => Promise<boolean>;
  setCurrentRoom: (room: Room | null) => void;
  
  // Message management
  loadMessages: (roomId: string, limit?: number, before?: string) => Promise<void>;
  sendMessage: (roomId: string, content: string, type?: 'text' | 'image' | 'file', replyToId?: string) => void;
  editMessage: (messageId: string, content: string) => Promise<boolean>;
  deleteMessage: (messageId: string) => Promise<boolean>;
  addMessage: (message: Message) => void;
  updateMessage: (message: Message) => void;
  removeMessage: (messageId: string, roomId: string) => void;
  updateMessageStatus: (messageId: string, status: 'sent' | 'delivered' | 'read') => void;
  
  // Reaction management
  addReaction: (messageId: string, emoji: string) => Promise<boolean>;
  removeReaction: (messageId: string, emoji: string) => Promise<boolean>;
  setMessageReactions: (messageId: string, reactions: MessageReaction[]) => void;
  
  // Thread management
  loadThread: (messageId: string) => Promise<void>;
  createThread: (messageId: string, initialReply: string) => Promise<boolean>;
  replyToThread: (threadId: string, content: string) => Promise<boolean>;
  setSelectedThread: (thread: MessageThread | null) => void;
  addThreadMessage: (threadId: string, message: Message) => void;
  
  // Custom emoji management
  loadRoomEmojis: (roomId: string) => Promise<void>;
  loadGlobalEmojis: () => Promise<void>;
  createCustomEmoji: (roomId: string, name: string, imageUrl: string, tags?: string[]) => Promise<boolean>;
  deleteCustomEmoji: (roomId: string, emojiId: string) => Promise<boolean>;
  
  // Spam detection management
  checkMessageSpam: (message: string) => Promise<any>;
  reportSpam: (message: string, isSpam: boolean) => Promise<boolean>;
  getSpamStats: () => Promise<void>;
  checkSpamServiceHealth: () => Promise<void>;
  
  // Offline message management
  loadOfflineMessages: () => Promise<void>;
  markOfflineMessagesReceived: (messageIds: string[]) => Promise<void>;
  addOfflineMessage: (message: OfflineMessage) => void;
  
  // Real-time features
  startTyping: (roomId: string) => void;
  stopTyping: (roomId: string) => void;
  setTypingUsers: (roomId: string, userIds: string[]) => void;
  setUserOnline: (userId: string, isOnline: boolean) => void;
  
  // Search
  searchMessages: (query: string, roomId?: string) => Promise<void>;
  clearSearchResults: () => void;
  
  // UI state management
  setEditingMessage: (message: Message | null) => void;
  setReplyingToMessage: (message: Message | null) => void;
  
  // Initialization
  initialize: () => void;
  cleanup: () => void;

  // Additional missing functions
  setSearchQuery: (query: string) => void;
  updateRoomSettings: (roomId: string, settings: any) => Promise<boolean>;
}

export const useChatStore = create<ChatStore>((set, get) => ({
  // Basic chat state
  rooms: [],
  messages: {},
  currentRoom: null,
  typingUsers: {},
  onlineUsers: {},
  
  // Extended state
  messageReactions: {},
  messageThreads: {},
  threadMessages: {},
  roomEmojis: {},
  globalEmojis: [],
  offlineMessages: [],
  searchResults: [],
  searchQuery: '',
  selectedThread: null,
  editingMessage: null,
  replyingToMessage: null,
  spamDetectionEnabled: false,
  spamStats: null,
  isSpamServiceHealthy: true,

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
        msg.id === message.id ? { ...msg, delivery_status: 'sent' as const } : msg
      );
      
      set({
        messages: {
          ...state.messages,
          [message.room_id]: updatedMessages,
        },
      });
    });

    // Message editing and deletion
    webSocketService.on('message_edited', (message: Message) => {
      get().updateMessage(message);
    });

    webSocketService.on('message_deleted', (data: { message_id: string; room_id: string }) => {
      get().removeMessage(data.message_id, data.room_id);
    });

    // Reaction events
    webSocketService.on('reaction_added', (data: { message_id: string; reaction: MessageReaction }) => {
      const state = get();
      const currentReactions = state.messageReactions[data.message_id] || [];
      const existingReaction = currentReactions.find(r => r.emoji === data.reaction.emoji);
      
      let updatedReactions;
      if (existingReaction) {
        updatedReactions = currentReactions.map(r =>
          r.emoji === data.reaction.emoji 
            ? { ...r, count: r.count + 1, users: [...r.users, ...data.reaction.users] }
            : r
        );
      } else {
        updatedReactions = [...currentReactions, data.reaction];
      }

      set({
        messageReactions: {
          ...state.messageReactions,
          [data.message_id]: updatedReactions,
        },
      });
    });

    webSocketService.on('reaction_removed', (data: { message_id: string; emoji: string; user_id: string }) => {
      const state = get();
      const currentReactions = state.messageReactions[data.message_id] || [];
      const updatedReactions = currentReactions.map(r => {
        if (r.emoji === data.emoji) {
          const updatedUsers = r.users.filter(userId => userId !== data.user_id);
          return updatedUsers.length > 0 
            ? { ...r, count: r.count - 1, users: updatedUsers }
            : null;
        }
        return r;
      }).filter(Boolean) as MessageReaction[];

      set({
        messageReactions: {
          ...state.messageReactions,
          [data.message_id]: updatedReactions,
        },
      });
    });

    // Thread events
    webSocketService.on('thread_created', (thread: MessageThread) => {
      const state = get();
      set({
        messageThreads: {
          ...state.messageThreads,
          [thread.parent_message_id]: thread,
        },
      });
    });

    webSocketService.on('thread_message_sent', (message: Message) => {
      if (message.thread_id) {
        get().addThreadMessage(message.thread_id, message);
      }
    });

    // Custom emoji events
    webSocketService.on('custom_emoji_added', (emoji: CustomEmoji) => {
      const state = get();
      if (emoji.room_id) {
        const currentEmojis = state.roomEmojis[emoji.room_id] || [];
        set({
          roomEmojis: {
            ...state.roomEmojis,
            [emoji.room_id]: [...currentEmojis, emoji],
          },
        });
      } else {
        // Global emoji
        set({
          globalEmojis: [...state.globalEmojis, emoji],
        });
      }
    });

    webSocketService.on('custom_emoji_deleted', (data: { emoji_id: string; room_id?: string }) => {
      const state = get();
      if (data.room_id) {
        const currentEmojis = state.roomEmojis[data.room_id] || [];
        const filteredEmojis = currentEmojis.filter(emoji => emoji.id !== data.emoji_id);
        set({
          roomEmojis: {
            ...state.roomEmojis,
            [data.room_id]: filteredEmojis,
          },
        });
      } else {
        // Global emoji
        const filteredEmojis = state.globalEmojis.filter(emoji => emoji.id !== data.emoji_id);
        set({ globalEmojis: filteredEmojis });
      }
    });

    // Offline message events
    webSocketService.on('offline_message', (message: OfflineMessage) => {
      get().addOfflineMessage(message);
    });

    // Search results
    webSocketService.on('search_results', (data: { messages: Message[]; query: string }) => {
      set({
        searchResults: data.messages,
        searchQuery: data.query,
      });
    });

    // Presence and activity updates
    webSocketService.on('presence_update', (data: { user_id: string; status: string }) => {
      get().setUserOnline(data.user_id, data.status === 'online');
    });

    webSocketService.on('activity_update', (data: any) => {
      // Handle activity updates if needed
      console.log('Activity update:', data);
    });

    // Sync events
    webSocketService.on('sync_state_updated', (data: any) => {
      // Handle multi-device sync updates
      console.log('Sync state updated:', data);
    });

    // Analytics events
    webSocketService.on('analytics_event', (data: any) => {
      // Handle analytics events if needed
      console.log('Analytics event:', data);
    });

    // Error handling
    webSocketService.on('error', (error: any) => {
      console.error('WebSocket error:', error);
    });
  },

  cleanup: () => {
    // Remove WebSocket listeners
    webSocketService.off('message', () => {});
    webSocketService.off('typing', () => {});
    webSocketService.off('presence', () => {});
    webSocketService.off('delivery_receipt', () => {});
    webSocketService.off('message_sent', () => {});
    webSocketService.off('message_edited', () => {});
    webSocketService.off('message_deleted', () => {});
    webSocketService.off('reaction_added', () => {});
    webSocketService.off('reaction_removed', () => {});
    webSocketService.off('thread_created', () => {});
    webSocketService.off('thread_message_sent', () => {});
    webSocketService.off('custom_emoji_added', () => {});
    webSocketService.off('custom_emoji_deleted', () => {});
    webSocketService.off('offline_message', () => {});
    webSocketService.off('search_results', () => {});
    webSocketService.off('presence_update', () => {});
    webSocketService.off('activity_update', () => {});
    webSocketService.off('sync_state_updated', () => {});
    webSocketService.off('analytics_event', () => {});
    webSocketService.off('error', () => {});
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
    const now = new Date().toISOString();
    const optimisticMessage: Message = {
      id: `temp-${Date.now()}`,
      sender_id: '', // Will be set by auth store
      room_id: roomId,
      content,
      type: type as 'text' | 'image' | 'file',
      timestamp: now,
      inserted_at: now,
      reply_to_id: replyToId,
      delivery_status: 'sent' as const,
    };

    // Add to local state immediately
    get().addMessage(optimisticMessage);

    // Send via WebSocket
    webSocketService.sendMessage(roomId, content, type as any, replyToId);
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

  searchMessages: async (query: string, roomId?: string): Promise<void> => {
    try {
      const response = await apiClient.searchMessages(query, roomId);
      if (response.success && response.data) {
        set({ 
          searchResults: response.data.messages,
          searchQuery: query 
        });
      }
    } catch (error) {
      console.error('Failed to search messages:', error);
      set({ searchResults: [], searchQuery: query });
    }
  },

  clearSearchResults: () => {
    set({ searchResults: [], searchQuery: '' });
  },

  // New message management methods
  editMessage: async (messageId: string, content: string): Promise<boolean> => {
    try {
      const response = await apiClient.editMessage(messageId, content);
      if (response.success && response.data) {
        get().updateMessage(response.data);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to edit message:', error);
      return false;
    }
  },

  deleteMessage: async (messageId: string): Promise<boolean> => {
    try {
      const response = await apiClient.deleteMessage(messageId);
      if (response.success) {
        // Find the message to get room ID
        const state = get();
        let roomId = '';
        Object.entries(state.messages).forEach(([rid, messages]) => {
          if (messages.find(m => m.id === messageId)) {
            roomId = rid;
          }
        });
        if (roomId) {
          get().removeMessage(messageId, roomId);
        }
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to delete message:', error);
      return false;
    }
  },

  updateMessage: (message: Message) => {
    const state = get();
    const roomMessages = state.messages[message.room_id] || [];
    const updatedMessages = roomMessages.map(msg =>
      msg.id === message.id ? message : msg
    );

    set({
      messages: {
        ...state.messages,
        [message.room_id]: updatedMessages,
      },
    });
  },

  removeMessage: (messageId: string, roomId: string) => {
    const state = get();
    const roomMessages = state.messages[roomId] || [];
    const filteredMessages = roomMessages.filter(msg => msg.id !== messageId);

    set({
      messages: {
        ...state.messages,
        [roomId]: filteredMessages,
      },
    });
  },

  // Reaction management
  addReaction: async (messageId: string, emoji: string): Promise<boolean> => {
    try {
      const response = await apiClient.addReaction(messageId, emoji);
      if (response.success && response.data) {
        const state = get();
        const currentReactions = state.messageReactions[messageId] || [];
        const existingReaction = currentReactions.find(r => r.emoji === emoji);
        
        let updatedReactions;
        if (existingReaction) {
          updatedReactions = currentReactions.map(r =>
            r.emoji === emoji ? { ...r, count: r.count + 1 } : r
          );
        } else {
          updatedReactions = [...currentReactions, response.data];
        }

        set({
          messageReactions: {
            ...state.messageReactions,
            [messageId]: updatedReactions,
          },
        });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to add reaction:', error);
      return false;
    }
  },

  removeReaction: async (messageId: string, emoji: string): Promise<boolean> => {
    try {
      const response = await apiClient.removeReaction(messageId, emoji);
      if (response.success) {
        const state = get();
        const currentReactions = state.messageReactions[messageId] || [];
        const updatedReactions = currentReactions.filter(r => r.emoji !== emoji);

        set({
          messageReactions: {
            ...state.messageReactions,
            [messageId]: updatedReactions,
          },
        });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to remove reaction:', error);
      return false;
    }
  },

  setMessageReactions: (messageId: string, reactions: MessageReaction[]) => {
    const state = get();
    set({
      messageReactions: {
        ...state.messageReactions,
        [messageId]: reactions,
      },
    });
  },

  // Thread management
  loadThread: async (messageId: string): Promise<void> => {
    try {
      const response = await apiClient.getThread(messageId);
      if (response.success && response.data) {
        const state = get();
        set({
          messageThreads: {
            ...state.messageThreads,
            [messageId]: response.data,
          },
        });

        // Load thread messages
        const messagesResponse = await apiClient.getThreadMessages(messageId);
        if (messagesResponse.success && messagesResponse.data) {
          set({
            threadMessages: {
              ...get().threadMessages,
              [response.data.id]: messagesResponse.data.messages,
            },
          });
        }
      }
    } catch (error) {
      console.error('Failed to load thread:', error);
    }
  },

  createThread: async (messageId: string, initialReply: string): Promise<boolean> => {
    try {
      webSocketService.createThread(messageId, initialReply);
      return true;
    } catch (error) {
      console.error('Failed to create thread:', error);
      return false;
    }
  },

  replyToThread: async (threadId: string, content: string): Promise<boolean> => {
    try {
      webSocketService.replyToThread(threadId, content);
      return true;
    } catch (error) {
      console.error('Failed to reply to thread:', error);
      return false;
    }
  },

  setSelectedThread: (thread: MessageThread | null) => {
    set({ selectedThread: thread });
  },

  addThreadMessage: (threadId: string, message: Message) => {
    const state = get();
    const currentMessages = state.threadMessages[threadId] || [];
    set({
      threadMessages: {
        ...state.threadMessages,
        [threadId]: [...currentMessages, message],
      },
    });
  },

  // Custom emoji management
  loadRoomEmojis: async (roomId: string): Promise<void> => {
    try {
      const response = await apiClient.getRoomEmojis(roomId);
      if (response.success && response.data) {
        const state = get();
        set({
          roomEmojis: {
            ...state.roomEmojis,
            [roomId]: response.data.emojis,
          },
        });
      }
    } catch (error) {
      console.error('Failed to load room emojis:', error);
    }
  },

  loadGlobalEmojis: async (): Promise<void> => {
    try {
      const response = await apiClient.getGlobalEmojis();
      if (response.success && response.data) {
        set({ globalEmojis: response.data.emojis });
      }
    } catch (error) {
      console.error('Failed to load global emojis:', error);
    }
  },

  createCustomEmoji: async (roomId: string, name: string, imageUrl: string, tags?: string[]): Promise<boolean> => {
    try {
      const response = await apiClient.createCustomEmoji(roomId, name, imageUrl, tags);
      if (response.success && response.data) {
        const state = get();
        const currentEmojis = state.roomEmojis[roomId] || [];
        set({
          roomEmojis: {
            ...state.roomEmojis,
            [roomId]: [...currentEmojis, response.data],
          },
        });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to create custom emoji:', error);
      return false;
    }
  },

  deleteCustomEmoji: async (emojiId: string): Promise<boolean> => {
    try {
      // Find the emoji to get the room ID
      const state = get();
      let roomId = '';
      Object.entries(state.roomEmojis).forEach(([rid, emojis]) => {
        if (emojis.find(e => e.id === emojiId)) {
          roomId = rid;
        }
      });

      if (roomId) {
        const response = await apiClient.deleteCustomEmoji(roomId, emojiId);
        if (response.success) {
          const currentEmojis = state.roomEmojis[roomId] || [];
          const filteredEmojis = currentEmojis.filter(emoji => emoji.id !== emojiId);
          set({
            roomEmojis: {
              ...state.roomEmojis,
              [roomId]: filteredEmojis,
            },
          });
          return true;
        }
      }
      return false;
    } catch (error) {
      console.error('Failed to delete custom emoji:', error);
      return false;
    }
  },

  // Offline message management
  loadOfflineMessages: async (): Promise<void> => {
    try {
      const response = await apiClient.getOfflineMessages();
      if (response.success && response.data) {
        set({ offlineMessages: response.data.messages });
      }
    } catch (error) {
      console.error('Failed to load offline messages:', error);
    }
  },

  markOfflineMessagesReceived: async (messageIds: string[]): Promise<void> => {
    try {
      const response = await apiClient.markOfflineMessagesReceived(messageIds);
      if (response.success) {
        const state = get();
        const remainingMessages = state.offlineMessages.filter(
          msg => !messageIds.includes(msg.id)
        );
        set({ offlineMessages: remainingMessages });
      }
    } catch (error) {
      console.error('Failed to mark offline messages as received:', error);
    }
  },

  addOfflineMessage: (message: OfflineMessage) => {
    const state = get();
    set({ offlineMessages: [...state.offlineMessages, message] });
  },

  // UI state management
  setEditingMessage: (message: Message | null) => {
    set({ editingMessage: message });
  },

  setReplyingToMessage: (message: Message | null) => {
    set({ replyingToMessage: message });
  },

  // Additional missing functions
  setSearchQuery: (query: string) => {
    set({ searchQuery: query });
  },

  updateRoomSettings: async (roomId: string, settings: any): Promise<boolean> => {
    try {
      // This would typically call an API endpoint to update room settings
      console.log('Update room settings:', { roomId, settings });
      return true;
    } catch (error) {
      console.error('Failed to update room settings:', error);
      return false;
    }
  },

  // Spam detection methods
  checkMessageSpam: async (message: string): Promise<any> => {
    try {
      const response = await apiClient.post('/api/spam/check', {
        message
      });
      
      if (response.ok) {
        const data = await response.json();
        return data.spam_check;
      } else {
        console.error('Failed to check spam');
        return null;
      }
    } catch (error) {
      console.error('Error checking spam:', error);
      return null;
    }
  },

  reportSpam: async (message: string, isSpam: boolean): Promise<boolean> => {
    try {
      const response = await apiClient.post('/api/spam/report', {
        message,
        is_spam: isSpam
      });

      if (response.ok) {
        return true;
      } else {
        console.error('Failed to report spam');
        return false;
      }
    } catch (error) {
      console.error('Error reporting spam:', error);
      return false;
    }
  },

  getSpamStats: async (): Promise<void> => {
    try {
      const response = await apiClient.get('/api/spam/stats');
      
      if (response.ok) {
        const data = await response.json();
        set({ spamStats: data.stats });
      } else {
        console.error('Failed to get spam stats');
      }
    } catch (error) {
      console.error('Error getting spam stats:', error);
    }
  },

  checkSpamServiceHealth: async (): Promise<void> => {
    try {
      const response = await apiClient.get('/api/spam/health');
      set({ isSpamServiceHealthy: response.ok });
    } catch (error) {
      console.error('Spam service health check failed:', error);
      set({ isSpamServiceHealthy: false });
    }
  },
}));
