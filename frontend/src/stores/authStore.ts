import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { User, AuthState } from '../types';
import { apiClient } from '../services/api';
import { webSocketService } from '../services/websocket';

interface AuthStore extends AuthState {
  login: (email: string, password: string) => Promise<boolean>;
  register: (email: string, username: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
  loadStoredAuth: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<boolean>;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthStore>((set, get) => ({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: false,

  setLoading: (loading: boolean) => {
    set({ isLoading: loading });
  },

  login: async (email: string, password: string): Promise<boolean> => {
    set({ isLoading: true });
    
    try {
      const response = await apiClient.login({ email, password });
      
      if (response.success && response.data) {
        const { user, token } = response.data;
        
        // Store credentials
        await AsyncStorage.setItem('auth_token', token);
        await AsyncStorage.setItem('user_data', JSON.stringify(user));
        
        // Update API client
        apiClient.setToken(token);
        webSocketService.setToken(token);
        
        set({
          user,
          token,
          isAuthenticated: true,
          isLoading: false,
        });
        
        // Connect to WebSocket
        try {
          await webSocketService.connect();
        } catch (wsError) {
          console.warn('WebSocket connection failed:', wsError);
        }
        
        return true;
      } else {
        set({ isLoading: false });
        return false;
      }
    } catch (error) {
      console.error('Login error:', error);
      set({ isLoading: false });
      return false;
    }
  },

  register: async (email: string, username: string, password: string): Promise<boolean> => {
    set({ isLoading: true });
    
    try {
      const response = await apiClient.register({ email, username, password });
      
      if (response.success && response.data) {
        const { user, token } = response.data;
        
        // Store credentials
        await AsyncStorage.setItem('auth_token', token);
        await AsyncStorage.setItem('user_data', JSON.stringify(user));
        
        // Update API client
        apiClient.setToken(token);
        webSocketService.setToken(token);
        
        set({
          user,
          token,
          isAuthenticated: true,
          isLoading: false,
        });
        
        // Connect to WebSocket
        try {
          await webSocketService.connect();
        } catch (wsError) {
          console.warn('WebSocket connection failed:', wsError);
        }
        
        return true;
      } else {
        set({ isLoading: false });
        return false;
      }
    } catch (error) {
      console.error('Registration error:', error);
      set({ isLoading: false });
      return false;
    }
  },

  logout: async (): Promise<void> => {
    try {
      // Disconnect WebSocket
      webSocketService.disconnect();
      
      // Clear stored data
      await AsyncStorage.removeItem('auth_token');
      await AsyncStorage.removeItem('user_data');
      
      // Clear API client token
      apiClient.setToken(null);
      webSocketService.setToken(null);
      
      set({
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
      });
    } catch (error) {
      console.error('Logout error:', error);
    }
  },

  loadStoredAuth: async (): Promise<void> => {
    set({ isLoading: true });
    
    try {
      const token = await AsyncStorage.getItem('auth_token');
      const userDataString = await AsyncStorage.getItem('user_data');
      
      if (token && userDataString) {
        const user = JSON.parse(userDataString);
        
        // Verify token is still valid
        apiClient.setToken(token);
        const profileResponse = await apiClient.getProfile();
        
        if (profileResponse.success && profileResponse.data) {
          webSocketService.setToken(token);
          
          set({
            user: profileResponse.data.user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          // Connect to WebSocket
          try {
            await webSocketService.connect();
          } catch (wsError) {
            console.warn('WebSocket connection failed:', wsError);
          }
        } else {
          // Token is invalid, clear stored data
          await get().logout();
        }
      } else {
        set({ isLoading: false });
      }
    } catch (error) {
      console.error('Load stored auth error:', error);
      set({ isLoading: false });
    }
  },

  updateProfile: async (updates: Partial<User>): Promise<boolean> => {
    try {
      const response = await apiClient.updateProfile(updates);
      
      if (response.success && response.data) {
        const updatedUser = response.data.user;
        
        // Update stored user data
        await AsyncStorage.setItem('user_data', JSON.stringify(updatedUser));
        
        set({ user: updatedUser });
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Update profile error:', error);
      return false;
    }
  },
}));
