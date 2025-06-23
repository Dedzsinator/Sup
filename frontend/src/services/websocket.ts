import { 
  WebSocketMessage, 
  Message, 
  TypingIndicator, 
  PresenceUpdate, 
  DeliveryReceipt 
} from '../types';

export type WebSocketEventCallback = (data: any) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private token: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, WebSocketEventCallback[]> = new Map();
  private isConnecting = false;

  constructor() {
    this.url = __DEV__ ? 'ws://localhost:4000/ws' : 'wss://your-production-url.com/ws';
  }

  setToken(token: string | null) {
    this.token = token;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
        resolve();
        return;
      }

      if (!this.token) {
        reject(new Error('No authentication token provided'));
        return;
      }

      this.isConnecting = true;
      
      try {
        this.ws = new WebSocket(`${this.url}?token=${this.token}`);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connected', null);
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.emit('disconnected', { code: event.code, reason: event.reason });
          
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
  }

  private scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (this.reconnectAttempts <= this.maxReconnectAttempts) {
        this.connect().catch(console.error);
      }
    }, delay);
  }

  private handleMessage(message: WebSocketMessage) {
    console.log('Received WebSocket message:', message);
    
    switch (message.type) {
      case 'message':
        this.emit('message', message.data as Message);
        break;
      case 'typing':
        this.emit('typing', message.data as TypingIndicator);
        break;
      case 'presence':
        this.emit('presence', message.data as PresenceUpdate);
        break;
      case 'delivery_receipt':
        this.emit('delivery_receipt', message.data as DeliveryReceipt);
        break;
      case 'message_sent':
        this.emit('message_sent', message.data as Message);
        break;
      case 'message_edited':
        this.emit('message_edited', message.data as Message);
        break;
      case 'message_deleted':
        this.emit('message_deleted', message.data as { message_id: string; room_id: string });
        break;
      case 'reaction_added':
        this.emit('reaction_added', message.data as { message_id: string; reaction: any });
        break;
      case 'reaction_removed':
        this.emit('reaction_removed', message.data as { message_id: string; emoji: string; user_id: string });
        break;
      case 'thread_created':
        this.emit('thread_created', message.data as any);
        break;
      case 'thread_message_sent':
        this.emit('thread_message_sent', message.data as Message);
        break;
      case 'custom_emoji_added':
        this.emit('custom_emoji_added', message.data as any);
        break;
      case 'custom_emoji_deleted':
        this.emit('custom_emoji_deleted', message.data as { emoji_id: string; room_id: string });
        break;
      case 'offline_message':
        this.emit('offline_message', message.data as any);
        break;
      case 'sync_state_updated':
        this.emit('sync_state_updated', message.data as any);
        break;
      case 'presence_update':
        this.emit('presence_update', message.data as { user_id: string; status: string; custom_status?: string });
        break;
      case 'activity_update':
        this.emit('activity_update', message.data as any);
        break;
      case 'call_initiated':
        this.emit('call_initiated', message.data as any);
        break;
      case 'call_answered':
        this.emit('call_answered', message.data as any);
        break;
      case 'call_declined':
        this.emit('call_declined', message.data as any);
        break;
      case 'call_ended':
        this.emit('call_ended', message.data as any);
        break;
      case 'webrtc_signaling':
        this.emit('webrtc_signaling', message.data as any);
        break;
      case 'analytics_event':
        this.emit('analytics_event', message.data as any);
        break;
      case 'mentions':
        this.emit('mentions', message.data as any);
        break;
      case 'search_results':
        this.emit('search_results', message.data as any);
        break;
      case 'conflict_resolved':
        this.emit('conflict_resolved', message.data as any);
        break;
      case 'error':
        this.emit('error', message.data);
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  // Event handling
  on(event: string, callback: WebSocketEventCallback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }

  off(event: string, callback: WebSocketEventCallback) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      const index = eventListeners.indexOf(callback);
      if (index > -1) {
        eventListeners.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => callback(data));
    }
  }

  private send(message: { type: string; data: any }) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  // Message sending methods
  sendMessage(roomId: string, content: string, type: 'text' | 'image' | 'file' | 'audio' | 'video' = 'text', replyToId?: string) {
    this.send({
      type: 'send_message',
      data: {
        room_id: roomId,
        content,
        type,
        reply_to_id: replyToId,
      },
    });
  }

  // Message editing and deletion
  editMessage(messageId: string, content: string) {
    this.send({
      type: 'edit_message',
      data: {
        message_id: messageId,
        content,
      },
    });
  }

  deleteMessage(messageId: string) {
    this.send({
      type: 'delete_message',
      data: {
        message_id: messageId,
      },
    });
  }

  // Reaction methods
  addReaction(messageId: string, emoji: string) {
    this.send({
      type: 'add_reaction',
      data: {
        message_id: messageId,
        emoji,
      },
    });
  }

  removeReaction(messageId: string, emoji: string) {
    this.send({
      type: 'remove_reaction',
      data: {
        message_id: messageId,
        emoji,
      },
    });
  }

  // Thread methods
  createThread(messageId: string, initialReply: string) {
    this.send({
      type: 'create_thread',
      data: {
        message_id: messageId,
        initial_reply: initialReply,
      },
    });
  }

  replyToThread(threadId: string, content: string, type: 'text' | 'image' | 'file' = 'text') {
    this.send({
      type: 'thread_reply',
      data: {
        thread_id: threadId,
        content,
        type,
      },
    });
  }

  // Typing indicators
  startTyping(roomId: string) {
    this.send({
      type: 'typing_start',
      data: { room_id: roomId },
    });
  }

  stopTyping(roomId: string) {
    this.send({
      type: 'typing_stop',
      data: { room_id: roomId },
    });
  }

  // Read receipts
  markMessageRead(messageId: string) {
    this.send({
      type: 'mark_read',
      data: { message_id: messageId },
    });
  }

  // Presence and status
  updatePresence(status: 'online' | 'away' | 'busy' | 'invisible') {
    this.send({
      type: 'update_presence',
      data: { status },
    });
  }

  updateActivity(activity: { type: string; name: string; details?: string }) {
    this.send({
      type: 'update_activity',
      data: { activity },
    });
  }

  // Room management
  joinRoom(roomId: string) {
    this.send({
      type: 'join_room',
      data: { room_id: roomId },
    });
  }

  leaveRoom(roomId: string) {
    this.send({
      type: 'leave_room',
      data: { room_id: roomId },
    });
  }

  // Custom emoji
  createCustomEmoji(roomId: string, name: string, imageUrl: string, tags?: string[]) {
    this.send({
      type: 'create_custom_emoji',
      data: {
        room_id: roomId,
        name,
        image_url: imageUrl,
        tags,
      },
    });
  }

  deleteCustomEmoji(roomId: string, emojiId: string) {
    this.send({
      type: 'delete_custom_emoji',
      data: {
        room_id: roomId,
        emoji_id: emojiId,
      },
    });
  }

  // Search
  searchMessages(query: string, roomId?: string, filters?: any) {
    this.send({
      type: 'search_messages',
      data: {
        query,
        room_id: roomId,
        filters,
      },
    });
  }

  // Voice/Video calls
  initiateCall(roomId: string, type: 'voice' | 'video', participants: string[]) {
    this.send({
      type: 'initiate_call',
      data: {
        room_id: roomId,
        call_type: type,
        participants,
      },
    });
  }

  answerCall(callId: string) {
    this.send({
      type: 'answer_call',
      data: { call_id: callId },
    });
  }

  declineCall(callId: string) {
    this.send({
      type: 'decline_call',
      data: { call_id: callId },
    });
  }

  endCall(callId: string) {
    this.send({
      type: 'end_call',
      data: { call_id: callId },
    });
  }

  sendWebRTCSignal(callId: string, signal: any) {
    this.send({
      type: 'webrtc_signal',
      data: {
        call_id: callId,
        signal,
      },
    });
  }

  // Multi-device sync
  requestSync() {
    this.send({
      type: 'request_sync',
      data: {},
    });
  }

  syncDeviceState(deviceInfo: any) {
    this.send({
      type: 'sync_device_state',
      data: { device_info: deviceInfo },
    });
  }

  // Offline message handling
  requestOfflineMessages() {
    this.send({
      type: 'request_offline_messages',
      data: {},
    });
  }

  acknowledgeMissedMessages(messageIds: string[]) {
    this.send({
      type: 'acknowledge_missed_messages',
      data: { message_ids: messageIds },
    });
  }

  // Analytics events
  trackEvent(eventType: string, eventData: any) {
    this.send({
      type: 'track_event',
      data: {
        event_type: eventType,
        event_data: eventData,
      },
    });
  }
}

export const webSocketService = new WebSocketService();
