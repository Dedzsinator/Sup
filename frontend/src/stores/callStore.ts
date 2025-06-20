import { create } from 'zustand';
import { Call, CallState, InitiateCallRequest } from '../types';
import { apiClient } from '../services/api';

interface CallStore extends CallState {
  initiateCall: (request: InitiateCallRequest) => Promise<boolean>;
  answerCall: (callId: string) => Promise<boolean>;
  declineCall: (callId: string) => Promise<boolean>;
  endCall: (callId: string) => Promise<boolean>;
  joinCall: (callId: string) => Promise<boolean>;
  leaveCall: (callId: string) => Promise<boolean>;
  
  // WebRTC methods
  setupLocalStream: (video: boolean, audio: boolean) => Promise<boolean>;
  toggleAudio: () => void;
  toggleVideo: () => void;
  toggleScreenShare: () => Promise<boolean>;
  
  // Stream management
  addRemoteStream: (userId: string, stream: MediaStream) => void;
  removeRemoteStream: (userId: string) => void;
  
  // Call history
  loadCallHistory: () => Promise<void>;
  
  // Cleanup
  cleanup: () => void;
}

export const useCallStore = create<CallStore>((set, get) => ({
  currentCall: null,
  incomingCall: null,
  callHistory: [],
  isCallActive: false,
  localStream: null,
  remoteStreams: {},
  audioEnabled: true,
  videoEnabled: true,
  screenShareEnabled: false,

  initiateCall: async (request: InitiateCallRequest): Promise<boolean> => {
    try {
      const response = await apiClient.initiateCall(request);
      
      if (response.success && response.data) {
        set({ 
          currentCall: response.data.call,
          isCallActive: true 
        });
        
        // Setup local stream
        await get().setupLocalStream(request.type === 'video', true);
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Initiate call error:', error);
      return false;
    }
  },

  answerCall: async (callId: string): Promise<boolean> => {
    try {
      const response = await apiClient.answerCall(callId);
      
      if (response.success && response.data) {
        const call = response.data.call;
        
        set({ 
          currentCall: call,
          incomingCall: null,
          isCallActive: true 
        });
        
        // Setup local stream
        await get().setupLocalStream(call.type === 'video', true);
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Answer call error:', error);
      return false;
    }
  },

  declineCall: async (callId: string): Promise<boolean> => {
    try {
      const response = await apiClient.declineCall(callId);
      
      if (response.success) {
        set({ 
          incomingCall: null,
          currentCall: null,
          isCallActive: false 
        });
        
        // Cleanup streams
        get().cleanup();
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Decline call error:', error);
      return false;
    }
  },

  endCall: async (callId: string): Promise<boolean> => {
    try {
      const response = await apiClient.endCall(callId);
      
      if (response.success) {
        set({ 
          currentCall: null,
          incomingCall: null,
          isCallActive: false 
        });
        
        // Cleanup streams
        get().cleanup();
        
        // Reload call history
        await get().loadCallHistory();
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('End call error:', error);
      return false;
    }
  },

  joinCall: async (callId: string): Promise<boolean> => {
    try {
      const response = await apiClient.joinCall(callId);
      
      if (response.success && response.data) {
        set({ 
          currentCall: response.data.call,
          isCallActive: true 
        });
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Join call error:', error);
      return false;
    }
  },

  leaveCall: async (callId: string): Promise<boolean> => {
    try {
      const response = await apiClient.leaveCall(callId);
      
      if (response.success) {
        set({ 
          currentCall: null,
          isCallActive: false 
        });
        
        // Cleanup streams
        get().cleanup();
        
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Leave call error:', error);
      return false;
    }
  },

  setupLocalStream: async (video: boolean, audio: boolean): Promise<boolean> => {
    try {
      const constraints = {
        audio: audio,
        video: video ? {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } : false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      set({ 
        localStream: stream,
        audioEnabled: audio,
        videoEnabled: video 
      });
      
      return true;
    } catch (error) {
      console.error('Setup local stream error:', error);
      return false;
    }
  },

  toggleAudio: () => {
    const state = get();
    const { localStream, audioEnabled } = state;
    
    if (localStream) {
      localStream.getAudioTracks().forEach(track => {
        track.enabled = !audioEnabled;
      });
      
      set({ audioEnabled: !audioEnabled });
    }
  },

  toggleVideo: () => {
    const state = get();
    const { localStream, videoEnabled } = state;
    
    if (localStream) {
      localStream.getVideoTracks().forEach(track => {
        track.enabled = !videoEnabled;
      });
      
      set({ videoEnabled: !videoEnabled });
    }
  },

  toggleScreenShare: async (): Promise<boolean> => {
    const state = get();
    const { screenShareEnabled, localStream } = state;
    
    try {
      if (!screenShareEnabled) {
        // Start screen share
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true
        });
        
        // Replace video track with screen share
        if (localStream) {
          const videoTrack = screenStream.getVideoTracks()[0];
          const sender = null; // You'd get this from your RTCPeerConnection
          
          // Replace track in peer connection
          // await sender.replaceTrack(videoTrack);
        }
        
        set({ screenShareEnabled: true });
        return true;
      } else {
        // Stop screen share and return to camera
        if (localStream) {
          const videoStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' }
          });
          
          const videoTrack = videoStream.getVideoTracks()[0];
          // Replace track back to camera
          // await sender.replaceTrack(videoTrack);
        }
        
        set({ screenShareEnabled: false });
        return true;
      }
    } catch (error) {
      console.error('Toggle screen share error:', error);
      return false;
    }
  },

  addRemoteStream: (userId: string, stream: MediaStream) => {
    const state = get();
    set({
      remoteStreams: {
        ...state.remoteStreams,
        [userId]: stream
      }
    });
  },

  removeRemoteStream: (userId: string) => {
    const state = get();
    const { [userId]: removed, ...remainingStreams } = state.remoteStreams;
    
    // Stop the removed stream
    if (removed) {
      removed.getTracks().forEach(track => track.stop());
    }
    
    set({ remoteStreams: remainingStreams });
  },

  loadCallHistory: async (): Promise<void> => {
    try {
      const response = await apiClient.getCallHistory();
      
      if (response.success && response.data) {
        set({ callHistory: response.data.calls });
      }
    } catch (error) {
      console.error('Load call history error:', error);
    }
  },

  cleanup: () => {
    const state = get();
    
    // Stop local stream
    if (state.localStream) {
      state.localStream.getTracks().forEach(track => track.stop());
    }
    
    // Stop all remote streams
    Object.values(state.remoteStreams).forEach(stream => {
      stream.getTracks().forEach(track => track.stop());
    });
    
    set({
      localStream: null,
      remoteStreams: {},
      audioEnabled: true,
      videoEnabled: true,
      screenShareEnabled: false
    });
  },
}));
