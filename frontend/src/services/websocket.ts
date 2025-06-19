import { WebSocketMessage, Message, TypingIndicator, PresenceUpdate, DeliveryReceipt } from '../types';

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

  markMessageRead(messageId: string) {
    this.send({
      type: 'mark_read',
      data: { message_id: messageId },
    });
  }

  joinRoom(roomId: string) {
    this.send({
      type: 'join_room',
      data: { room_id: roomId },
    });
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
}

export const webSocketService = new WebSocketService();
