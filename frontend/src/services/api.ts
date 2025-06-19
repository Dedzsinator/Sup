import { ApiResponse, RegisterRequest, LoginRequest, User } from '../types';

const API_BASE_URL = __DEV__ ? 'http://localhost:4000' : 'https://your-production-url.com';

class ApiClient {
  private baseURL: string;
  private token: string | null = null;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  setToken(token: string | null) {
    this.token = token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: data.error || 'Request failed',
        };
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
      };
    }
  }

  // Authentication endpoints
  async register(params: RegisterRequest): Promise<ApiResponse<{ user: User; token: string }>> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async login(params: LoginRequest): Promise<ApiResponse<{ user: User; token: string }>> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // User endpoints
  async getProfile(): Promise<ApiResponse<{ user: User }>> {
    return this.request('/api/profile');
  }

  async updateProfile(updates: Partial<User>): Promise<ApiResponse<{ user: User }>> {
    return this.request('/api/profile', {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  // Room endpoints
  async getRooms(): Promise<ApiResponse<{ rooms: any[] }>> {
    return this.request('/api/rooms');
  }

  async createRoom(params: any): Promise<ApiResponse<any>> {
    return this.request('/api/rooms', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async joinRoom(roomId: string): Promise<ApiResponse<any>> {
    return this.request(`/api/rooms/${roomId}/join`, {
      method: 'POST',
    });
  }

  async leaveRoom(roomId: string): Promise<ApiResponse<any>> {
    return this.request(`/api/rooms/${roomId}/leave`, {
      method: 'DELETE',
    });
  }

  async getRoomMessages(roomId: string, limit = 50, before?: string): Promise<ApiResponse<{ messages: any[] }>> {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (before) params.set('before', before);
    
    return this.request(`/api/rooms/${roomId}/messages?${params}`);
  }

  async createDirectMessage(userId: string): Promise<ApiResponse<any>> {
    return this.request('/api/direct_messages', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId }),
    });
  }

  // Search endpoints
  async searchMessages(query: string, limit = 20): Promise<ApiResponse<{ messages: any[] }>> {
    const params = new URLSearchParams({ q: query, limit: limit.toString() });
    return this.request(`/api/search/messages?${params}`);
  }

  // Push notification endpoints
  async registerPushToken(token: string, platform: string): Promise<ApiResponse<any>> {
    return this.request('/api/push/register', {
      method: 'POST',
      body: JSON.stringify({ token, platform }),
    });
  }

  async unregisterPushToken(): Promise<ApiResponse<any>> {
    return this.request('/api/push/unregister', {
      method: 'DELETE',
    });
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
