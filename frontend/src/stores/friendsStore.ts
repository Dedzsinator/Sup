import { create } from 'zustand';
import { User, FriendRequest, FriendsState } from '../types';
import { apiClient } from '../services/api';

interface FriendsStore extends FriendsState {
  sendFriendRequest: (identifier: string) => Promise<boolean>;
  acceptFriendRequest: (requestId: string) => Promise<boolean>;
  declineFriendRequest: (requestId: string) => Promise<boolean>;
  removeFriend: (friendId: string) => Promise<boolean>;
  blockUser: (userId: string) => Promise<boolean>;
  unblockUser: (userId: string) => Promise<boolean>;
  loadFriends: () => Promise<void>;
  loadFriendRequests: () => Promise<void>;
  loadBlockedUsers: () => Promise<void>;
  searchUsers: (query: string) => Promise<User[]>;
}

export const useFriendsStore = create<FriendsStore>((set, get) => ({
  friends: [],
  friendRequests: [],
  sentRequests: [],
  blockedUsers: [],
  isLoading: false,

  sendFriendRequest: async (identifier: string): Promise<boolean> => {
    try {
      const response = await apiClient.sendFriendRequest(identifier);
      
      if (response.success) {
        // Refresh sent requests
        await get().loadFriendRequests();
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Send friend request error:', error);
      return false;
    }
  },

  acceptFriendRequest: async (requestId: string): Promise<boolean> => {
    try {
      const response = await apiClient.acceptFriendRequest(requestId);
      
      if (response.success) {
        // Refresh friends and requests
        await Promise.all([
          get().loadFriends(),
          get().loadFriendRequests()
        ]);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Accept friend request error:', error);
      return false;
    }
  },

  declineFriendRequest: async (requestId: string): Promise<boolean> => {
    try {
      const response = await apiClient.declineFriendRequest(requestId);
      
      if (response.success) {
        // Remove from local state
        const state = get();
        set({
          friendRequests: state.friendRequests.filter(req => req.id !== requestId)
        });
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Decline friend request error:', error);
      return false;
    }
  },

  removeFriend: async (friendId: string): Promise<boolean> => {
    try {
      const response = await apiClient.removeFriend(friendId);
      
      if (response.success) {
        // Remove from local state
        const state = get();
        set({
          friends: state.friends.filter(friend => friend.id !== friendId)
        });
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Remove friend error:', error);
      return false;
    }
  },

  blockUser: async (userId: string): Promise<boolean> => {
    try {
      const response = await apiClient.blockUser(userId);
      
      if (response.success) {
        await Promise.all([
          get().loadFriends(),
          get().loadBlockedUsers()
        ]);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Block user error:', error);
      return false;
    }
  },

  unblockUser: async (userId: string): Promise<boolean> => {
    try {
      const response = await apiClient.unblockUser(userId);
      
      if (response.success) {
        // Remove from local state
        const state = get();
        set({
          blockedUsers: state.blockedUsers.filter(user => user.id !== userId)
        });
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Unblock user error:', error);
      return false;
    }
  },

  loadFriends: async (): Promise<void> => {
    set({ isLoading: true });
    
    try {
      const response = await apiClient.getFriends();
      
      if (response.success && response.data) {
        set({ 
          friends: response.data.friends,
          isLoading: false 
        });
      } else {
        set({ isLoading: false });
      }
    } catch (error) {
      console.error('Load friends error:', error);
      set({ isLoading: false });
    }
  },

  loadFriendRequests: async (): Promise<void> => {
    try {
      const [receivedResponse, sentResponse] = await Promise.all([
        apiClient.getFriendRequests(),
        apiClient.getSentFriendRequests()
      ]);
      
      if (receivedResponse.success && receivedResponse.data) {
        set({ friendRequests: receivedResponse.data.requests });
      }
      
      if (sentResponse.success && sentResponse.data) {
        set({ sentRequests: sentResponse.data.requests });
      }
    } catch (error) {
      console.error('Load friend requests error:', error);
    }
  },

  loadBlockedUsers: async (): Promise<void> => {
    try {
      const response = await apiClient.getBlockedUsers();
      
      if (response.success && response.data) {
        set({ blockedUsers: response.data.users });
      }
    } catch (error) {
      console.error('Load blocked users error:', error);
    }
  },

  searchUsers: async (query: string): Promise<User[]> => {
    try {
      const response = await apiClient.searchUsers(query);
      
      if (response.success && response.data) {
        return response.data.users;
      }
      
      return [];
    } catch (error) {
      console.error('Search users error:', error);
      return [];
    }
  },
}));
