import { 
  ApiResponse, 
  RegisterRequest, 
  LoginRequest, 
  User, 
  UpdateProfileRequest,
  UpdateSettingsRequest,
  Call,
  FriendRequest,
  UserSettings
} from '../types';

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

  private getHeaders() {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    return headers;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
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

  async updateProfile(updates: UpdateProfileRequest): Promise<ApiResponse<{ user: User }>> {
    return this.request('/api/profile', {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async updateSettings(settings: UpdateSettingsRequest): Promise<ApiResponse<{ settings: UserSettings }>> {
    return this.request('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    });
  }

  async getSettings(): Promise<ApiResponse<{ settings: UserSettings }>> {
    return this.request('/api/settings');
  }

  // Friend management endpoints
  async getFriends(): Promise<{ friends: User[] }> {
    const response = await fetch(`${this.baseURL}/api/friends`, {
      method: 'GET',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get friends: ${response.statusText}`);
    }

    return response.json();
  }

  async getFriendRequests(): Promise<{ requests: FriendRequest[] }> {
    const response = await fetch(`${this.baseURL}/api/friends/requests`, {
      method: 'GET',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get friend requests: ${response.statusText}`);
    }

    return response.json();
  }

  async sendFriendRequest(targetUserId: string): Promise<{ request: FriendRequest }> {
    const response = await fetch(`${this.baseURL}/api/friends/request`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ target_user_id: targetUserId }),
    });

    if (!response.ok) {
      throw new Error(`Failed to send friend request: ${response.statusText}`);
    }

    return response.json();
  }

  async respondToFriendRequest(requestId: string, action: 'accept' | 'reject'): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/friends/request/${requestId}`, {
      method: 'PUT',
      headers: this.getHeaders(),
      body: JSON.stringify({ action }),
    });

    if (!response.ok) {
      throw new Error(`Failed to respond to friend request: ${response.statusText}`);
    }

    return response.json();
  }

  async removeFriend(friendId: string): Promise<{ message: string }> {
    const response = await fetch(`${this.baseURL}/api/friends/${friendId}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to remove friend: ${response.statusText}`);
    }

    return response.json();
  }

  async blockUser(targetUserId: string): Promise<{ message: string }> {
    const response = await fetch(`${this.baseURL}/api/friends/block`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ target_user_id: targetUserId }),
    });

    if (!response.ok) {
      throw new Error(`Failed to block user: ${response.statusText}`);
    }

    return response.json();
  }

  async unblockUser(blockedUserId: string): Promise<{ message: string }> {
    const response = await fetch(`${this.baseURL}/api/friends/block/${blockedUserId}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to unblock user: ${response.statusText}`);
    }

    return response.json();
  }

  async getBlockedUsers(): Promise<{ blocked_users: User[] }> {
    const response = await fetch(`${this.baseURL}/api/friends/blocked`, {
      method: 'GET',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get blocked users: ${response.statusText}`);
    }

    return response.json();
  }

  async searchUsers(query: string, limit: number = 10): Promise<{ users: User[] }> {
    const params = new URLSearchParams({
      q: query,
      limit: limit.toString(),
    });

    const response = await fetch(`${this.baseURL}/api/users/search?${params}`, {
      method: 'GET',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to search users: ${response.statusText}`);
    }

    return response.json();
  }

  // Call management methods
  async initiateCall(targetUserId: string, callType: 'voice' | 'video'): Promise<{ call: Call }> {
    const response = await fetch(`${this.baseURL}/api/calls`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        target_user_id: targetUserId,
        call_type: callType,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to initiate call: ${response.statusText}`);
    }

    return response.json();
  }

  async acceptCall(callId: string): Promise<{ call: Call }> {
    const response = await fetch(`${this.baseURL}/api/calls/${callId}/accept`, {
      method: 'PUT',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to accept call: ${response.statusText}`);
    }

    return response.json();
  }

  async rejectCall(callId: string): Promise<{ call: Call }> {
    const response = await fetch(`${this.baseURL}/api/calls/${callId}/reject`, {
      method: 'PUT',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to reject call: ${response.statusText}`);
    }

    return response.json();
  }

  async endCall(callId: string): Promise<{ call: Call }> {
    const response = await fetch(`${this.baseURL}/api/calls/${callId}/end`, {
      method: 'PUT',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to end call: ${response.statusText}`);
    }

    return response.json();
  }

  async sendWebRTCSignal(callId: string, signal: any): Promise<{ message: string }> {
    const response = await fetch(`${this.baseURL}/api/calls/${callId}/signal`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ signal }),
    });

    if (!response.ok) {
      throw new Error(`Failed to send WebRTC signal: ${response.statusText}`);
    }

    return response.json();
  }

  async getActiveCalls(): Promise<{ calls: Call[] }> {
    const response = await fetch(`${this.baseURL}/api/calls/active`, {
      method: 'GET',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get active calls: ${response.statusText}`);
    }

    return response.json();
  }

  // File upload methods
  async uploadAvatar(file: File | FormData): Promise<{ avatar_url: string }> {
    const formData = file instanceof FormData ? file : new FormData();
    if (!(file instanceof FormData)) {
      formData.append('avatar', file);
    }

    const response = await fetch(`${this.baseURL}/api/upload/avatar`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
        // Don't set Content-Type for FormData, let the browser set it
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload avatar: ${response.statusText}`);
    }

    return response.json();
  }

  async uploadBanner(file: File | FormData): Promise<{ banner_url: string }> {
    const formData = file instanceof FormData ? file : new FormData();
    if (!(file instanceof FormData)) {
      formData.append('banner', file);
    }

    const response = await fetch(`${this.baseURL}/api/upload/banner`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
        // Don't set Content-Type for FormData, let the browser set it
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload banner: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
