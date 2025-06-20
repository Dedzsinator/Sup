import React, { useState, useEffect, useRef } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Animated, Dimensions, Alert } from 'react-native';
import { Portal, Modal, Button, Avatar } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, colors, Spacing } from '../../theme';
import { useCallStore } from '../../stores/callStore';
import { Call, CallStatus, User } from '../../types';
import ModernCard from '../../components/ModernCard';

const { width, height } = Dimensions.get('window');

interface CallScreenProps {
    call: Call;
    user: User;
    onEndCall: () => void;
}

export default function CallScreen({ call, user, onEndCall }: CallScreenProps) {
    const theme = useTheme();
    const {
        isMuted,
        isVideoEnabled,
        isScreenSharing,
        toggleMute,
        toggleVideo,
        toggleScreenShare,
        acceptCall,
        rejectCall,
        endCall
    } = useCallStore();

    const [callDuration, setCallDuration] = useState(0);
    const [showControls, setShowControls] = useState(true);
    const localVideoRef = useRef(null);
    const remoteVideoRef = useRef(null);
    const hideControlsTimeout = useRef<NodeJS.Timeout>();
    const pulseAnim = useRef(new Animated.Value(1)).current;

    useEffect(() => {
        if (call.status === 'active') {
            const interval = setInterval(() => {
                setCallDuration(prev => prev + 1);
            }, 1000);
            return () => clearInterval(interval);
        }
    }, [call.status]);

    useEffect(() => {
        // Auto-hide controls after 5 seconds of inactivity
        if (showControls && call.status === 'active') {
            hideControlsTimeout.current = setTimeout(() => {
                setShowControls(false);
            }, 5000);
        }

        return () => {
            if (hideControlsTimeout.current) {
                clearTimeout(hideControlsTimeout.current);
            }
        };
    }, [showControls, call.status]);

    useEffect(() => {
        // Pulse animation for incoming calls
        if (call.status === 'ringing') {
            const pulse = Animated.loop(
                Animated.sequence([
                    Animated.timing(pulseAnim, {
                        toValue: 1.2,
                        duration: 1000,
                        useNativeDriver: true,
                    }),
                    Animated.timing(pulseAnim, {
                        toValue: 1,
                        duration: 1000,
                        useNativeDriver: true,
                    }),
                ]),
            );
            pulse.start();
            return () => pulse.stop();
        }
    }, [call.status, pulseAnim]);

    const formatDuration = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const handleAcceptCall = async () => {
        try {
            await acceptCall(call.id);
        } catch (error) {
            console.error('Failed to accept call:', error);
            Alert.alert('Error', 'Failed to accept call');
        }
    };

    const handleRejectCall = async () => {
        try {
            await rejectCall(call.id);
            onEndCall();
        } catch (error) {
            console.error('Failed to reject call:', error);
            Alert.alert('Error', 'Failed to reject call');
        }
    };

    const handleEndCall = async () => {
        try {
            await endCall(call.id);
            onEndCall();
        } catch (error) {
            console.error('Failed to end call:', error);
            Alert.alert('Error', 'Failed to end call');
        }
    };

    const toggleControlsVisibility = () => {
        setShowControls(!showControls);
    };

    const otherParticipant = call.participants.find(p => p.userId !== user.id);

    const renderIncomingCall = () => (
        <View style={[styles.incomingCallContainer, { backgroundColor: theme.colors.surface }]}>
            <View style={styles.incomingCallContent}>
                <Animated.View style={[styles.avatarContainer, { transform: [{ scale: pulseAnim }] }]}>
                    <Avatar.Image
                        size={120}
                        source={{ uri: otherParticipant?.user.avatarUrl || 'https://via.placeholder.com/120' }}
                    />
                </Animated.View>

                <Text style={[styles.callerName, { color: theme.colors.onSurface }]}>
                    {otherParticipant?.user.displayName || otherParticipant?.user.username}
                </Text>

                <Text style={[styles.callType, { color: theme.colors.onSurfaceVariant }]}>
                    Incoming {call.callType} call
                </Text>

                <View style={styles.incomingCallActions}>
                    <TouchableOpacity
                        style={[styles.callButton, styles.rejectButton]}
                        onPress={handleRejectCall}
                    >
                        <Ionicons name="call" size={30} color="white" style={{ transform: [{ rotate: '135deg' }] }} />
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={[styles.callButton, styles.acceptButton]}
                        onPress={handleAcceptCall}
                    >
                        <Ionicons name="call" size={30} color="white" />
                    </TouchableOpacity>
                </View>
            </View>
        </View>
    );

    const renderActiveCall = () => (
        <TouchableOpacity
            style={styles.activeCallContainer}
            activeOpacity={1}
            onPress={toggleControlsVisibility}
        >
            {call.callType === 'video' && (
                <>
                    {/* Remote video */}
                    <View style={styles.remoteVideoContainer}>
                        {/* Placeholder for remote video stream */}
                        <View style={[styles.videoPlaceholder, { backgroundColor: colors.gray[800] }]}>
                            <Avatar.Image
                                size={80}
                                source={{ uri: otherParticipant?.user.avatarUrl || 'https://via.placeholder.com/80' }}
                            />
                            <Text style={[styles.videoPlaceholderText, { color: colors.gray[300] }]}>
                                {otherParticipant?.user.displayName || otherParticipant?.user.username}
                            </Text>
                        </View>
                    </View>

                    {/* Local video (picture-in-picture) */}
                    {isVideoEnabled && (
                        <View style={styles.localVideoContainer}>
                            <View style={[styles.localVideo, { backgroundColor: colors.gray[700] }]}>
                                <Text style={[styles.localVideoText, { color: colors.gray[300] }]}>You</Text>
                            </View>
                        </View>
                    )}
                </>
            )}

            {call.callType === 'voice' && (
                <View style={[styles.voiceCallContainer, { backgroundColor: theme.colors.surface }]}>
                    <Avatar.Image
                        size={100}
                        source={{ uri: otherParticipant?.user.avatarUrl || 'https://via.placeholder.com/100' }}
                    />
                    <Text style={[styles.callerName, { color: theme.colors.onSurface }]}>
                        {otherParticipant?.user.displayName || otherParticipant?.user.username}
                    </Text>
                    <Text style={[styles.callDuration, { color: theme.colors.onSurfaceVariant }]}>
                        {formatDuration(callDuration)}
                    </Text>
                </View>
            )}

            {/* Call controls */}
            {showControls && (
                <Animated.View
                    style={[
                        styles.controlsContainer,
                        { backgroundColor: 'rgba(0,0,0,0.7)' }
                    ]}
                >
                    <View style={styles.callInfo}>
                        <Text style={[styles.callDurationControl, { color: 'white' }]}>
                            {formatDuration(callDuration)}
                        </Text>
                    </View>

                    <View style={styles.controlsRow}>
                        <TouchableOpacity
                            style={[
                                styles.controlButton,
                                isMuted && styles.controlButtonActive
                            ]}
                            onPress={toggleMute}
                        >
                            <Ionicons
                                name={isMuted ? "mic-off" : "mic"}
                                size={24}
                                color="white"
                            />
                        </TouchableOpacity>

                        {call.callType === 'video' && (
                            <TouchableOpacity
                                style={[
                                    styles.controlButton,
                                    !isVideoEnabled && styles.controlButtonActive
                                ]}
                                onPress={toggleVideo}
                            >
                                <Ionicons
                                    name={isVideoEnabled ? "videocam" : "videocam-off"}
                                    size={24}
                                    color="white"
                                />
                            </TouchableOpacity>
                        )}

                        <TouchableOpacity
                            style={[
                                styles.controlButton,
                                isScreenSharing && styles.controlButtonActive
                            ]}
                            onPress={toggleScreenShare}
                        >
                            <Ionicons
                                name="phone-portrait"
                                size={24}
                                color="white"
                            />
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={[styles.controlButton, styles.endCallButton]}
                            onPress={handleEndCall}
                        >
                            <Ionicons name="call" size={24} color="white" style={{ transform: [{ rotate: '135deg' }] }} />
                        </TouchableOpacity>
                    </View>
                </Animated.View>
            )}
        </TouchableOpacity>
    );

    return (
        <Portal>
            <Modal
                visible={true}
                onDismiss={() => { }}
                contentContainerStyle={styles.modalContainer}
            >
                {call.status === 'ringing' && renderIncomingCall()}
                {(call.status === 'active' || call.status === 'connecting') && renderActiveCall()}
            </Modal>
        </Portal>
    );
}

const styles = StyleSheet.create({
    modalContainer: {
        flex: 1,
        backgroundColor: 'black',
    },
    incomingCallContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: Spacing.xl,
    },
    incomingCallContent: {
        alignItems: 'center',
        width: '100%',
    },
    avatarContainer: {
        marginBottom: Spacing.xl,
    },
    callerName: {
        fontSize: 28,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: Spacing.sm,
    },
    callType: {
        fontSize: 18,
        textAlign: 'center',
        marginBottom: Spacing.xl * 2,
    },
    incomingCallActions: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        width: '100%',
        maxWidth: 300,
    },
    callButton: {
        width: 70,
        height: 70,
        borderRadius: 35,
        justifyContent: 'center',
        alignItems: 'center',
        elevation: 4,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
    },
    acceptButton: {
        backgroundColor: colors.success[500],
    },
    rejectButton: {
        backgroundColor: colors.error[500],
    },
    activeCallContainer: {
        flex: 1,
        position: 'relative',
    },
    remoteVideoContainer: {
        flex: 1,
        backgroundColor: 'black',
    },
    videoPlaceholder: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    videoPlaceholderText: {
        fontSize: 18,
        fontWeight: '500',
        marginTop: Spacing.md,
    },
    localVideoContainer: {
        position: 'absolute',
        top: Spacing.xl,
        right: Spacing.lg,
        width: 120,
        height: 160,
        borderRadius: 12,
        overflow: 'hidden',
        elevation: 4,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
    },
    localVideo: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    localVideoText: {
        fontSize: 14,
        fontWeight: '500',
    },
    voiceCallContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: Spacing.xl,
    },
    callDuration: {
        fontSize: 18,
        marginTop: Spacing.md,
    },
    controlsContainer: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingVertical: Spacing.xl,
        paddingHorizontal: Spacing.lg,
    },
    callInfo: {
        alignItems: 'center',
        marginBottom: Spacing.lg,
    },
    callDurationControl: {
        fontSize: 16,
        fontWeight: '500',
    },
    controlsRow: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
    },
    controlButton: {
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: 'rgba(255,255,255,0.2)',
        justifyContent: 'center',
        alignItems: 'center',
        marginHorizontal: Spacing.md,
    },
    controlButtonActive: {
        backgroundColor: colors.error[500],
    },
    endCallButton: {
        backgroundColor: colors.error[500],
    },
});
