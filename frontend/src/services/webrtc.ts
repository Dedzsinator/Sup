import { RTCPeerConnection, RTCIceCandidate, RTCSessionDescription } from 'react-native-webrtc';
import { apiClient } from './api';

export interface WebRTCConfig {
    iceServers: RTCIceServer[];
}

export interface RTCIceServer {
    urls: string | string[];
    username?: string;
    credential?: string;
}

const DEFAULT_WEBRTC_CONFIG: WebRTCConfig = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
    ],
};

export class WebRTCService {
    private peerConnection: RTCPeerConnection | null = null;
    private localStream: MediaStream | null = null;
    private remoteStream: MediaStream | null = null;
    private callId: string | null = null;
    private isInitiator: boolean = false;

    private onRemoteStreamCallback?: (stream: MediaStream) => void;
    private onLocalStreamCallback?: (stream: MediaStream) => void;
    private onConnectionStateChangeCallback?: (state: string) => void;
    private onIceCandidateCallback?: (candidate: RTCIceCandidate) => void;

    constructor(config: WebRTCConfig = DEFAULT_WEBRTC_CONFIG) {
        this.initializePeerConnection(config);
    }

    private initializePeerConnection(config: WebRTCConfig) {
        this.peerConnection = new RTCPeerConnection(config);

        // Handle ICE candidates
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.callId) {
                this.sendSignal({
                    type: 'ice-candidate',
                    candidate: event.candidate,
                });
                this.onIceCandidateCallback?.(event.candidate);
            }
        };

        // Handle remote stream
        this.peerConnection.onaddstream = (event) => {
            this.remoteStream = event.stream;
            this.onRemoteStreamCallback?.(event.stream);
        };

        // Handle connection state changes
        this.peerConnection.onconnectionstatechange = () => {
            const state = this.peerConnection?.connectionState || 'unknown';
            console.log('WebRTC connection state:', state);
            this.onConnectionStateChangeCallback?.(state);
        };

        // Handle ICE connection state changes
        this.peerConnection.oniceconnectionstatechange = () => {
            const state = this.peerConnection?.iceConnectionState || 'unknown';
            console.log('WebRTC ICE connection state:', state);
        };
    }

    async initializeCall(callId: string, isInitiator: boolean = false, callType: 'voice' | 'video' = 'voice') {
        this.callId = callId;
        this.isInitiator = isInitiator;

        try {
            // Get user media
            this.localStream = await this.getUserMedia(callType);
            
            if (this.localStream && this.peerConnection) {
                this.peerConnection.addStream(this.localStream);
                this.onLocalStreamCallback?.(this.localStream);
            }

            if (isInitiator) {
                await this.createOffer();
            }
        } catch (error) {
            console.error('Failed to initialize call:', error);
            throw error;
        }
    }

    private async getUserMedia(callType: 'voice' | 'video'): Promise<MediaStream> {
        const constraints = {
            audio: true,
            video: callType === 'video' ? {
                width: { min: 640, ideal: 1280 },
                height: { min: 480, ideal: 720 },
                frameRate: { ideal: 30, max: 30 }
            } : false,
        };

        // Note: react-native-webrtc uses different method names
        const mediaDevices = require('react-native-webrtc').mediaDevices;
        return await mediaDevices.getUserMedia(constraints);
    }

    private async createOffer() {
        if (!this.peerConnection) return;

        try {
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            
            await this.sendSignal({
                type: 'offer',
                sdp: offer,
            });
        } catch (error) {
            console.error('Failed to create offer:', error);
            throw error;
        }
    }

    async handleOffer(offer: RTCSessionDescription) {
        if (!this.peerConnection) return;

        try {
            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
            const answer = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(answer);
            
            await this.sendSignal({
                type: 'answer',
                sdp: answer,
            });
        } catch (error) {
            console.error('Failed to handle offer:', error);
            throw error;
        }
    }

    async handleAnswer(answer: RTCSessionDescription) {
        if (!this.peerConnection) return;

        try {
            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
        } catch (error) {
            console.error('Failed to handle answer:', error);
            throw error;
        }
    }

    async handleIceCandidate(candidate: RTCIceCandidate) {
        if (!this.peerConnection) return;

        try {
            await this.peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
        } catch (error) {
            console.error('Failed to handle ICE candidate:', error);
        }
    }

    private async sendSignal(signal: any) {
        if (!this.callId) return;

        try {
            await apiClient.sendWebRTCSignal(this.callId, signal);
        } catch (error) {
            console.error('Failed to send signal:', error);
        }
    }

    toggleMute(): boolean {
        if (!this.localStream) return false;

        const audioTracks = this.localStream.getAudioTracks();
        if (audioTracks.length > 0) {
            const enabled = !audioTracks[0].enabled;
            audioTracks[0].enabled = enabled;
            return !enabled; // Return muted state
        }
        return false;
    }

    toggleVideo(): boolean {
        if (!this.localStream) return false;

        const videoTracks = this.localStream.getVideoTracks();
        if (videoTracks.length > 0) {
            const enabled = !videoTracks[0].enabled;
            videoTracks[0].enabled = enabled;
            return enabled; // Return video enabled state
        }
        return false;
    }

    async switchCamera() {
        if (!this.localStream) return;

        const videoTracks = this.localStream.getVideoTracks();
        if (videoTracks.length > 0) {
            // Note: Camera switching implementation depends on react-native-webrtc version
            try {
                const videoTrack = videoTracks[0];
                if (videoTrack._switchCamera) {
                    videoTrack._switchCamera();
                }
            } catch (error) {
                console.error('Failed to switch camera:', error);
            }
        }
    }

    getLocalStream(): MediaStream | null {
        return this.localStream;
    }

    getRemoteStream(): MediaStream | null {
        return this.remoteStream;
    }

    setOnRemoteStream(callback: (stream: MediaStream) => void) {
        this.onRemoteStreamCallback = callback;
    }

    setOnLocalStream(callback: (stream: MediaStream) => void) {
        this.onLocalStreamCallback = callback;
    }

    setOnConnectionStateChange(callback: (state: string) => void) {
        this.onConnectionStateChangeCallback = callback;
    }

    setOnIceCandidate(callback: (candidate: RTCIceCandidate) => void) {
        this.onIceCandidateCallback = callback;
    }

    async cleanup() {
        // Stop local stream tracks
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }

        // Close peer connection
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }

        // Reset state
        this.remoteStream = null;
        this.callId = null;
        this.isInitiator = false;
        
        // Clear callbacks
        this.onRemoteStreamCallback = undefined;
        this.onLocalStreamCallback = undefined;
        this.onConnectionStateChangeCallback = undefined;
        this.onIceCandidateCallback = undefined;
    }

    getConnectionState(): string {
        return this.peerConnection?.connectionState || 'unknown';
    }

    getIceConnectionState(): string {
        return this.peerConnection?.iceConnectionState || 'unknown';
    }
}

// Singleton instance
export const webRTCService = new WebRTCService();
