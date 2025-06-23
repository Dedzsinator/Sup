import React, { useEffect, useState, useRef, useCallback } from 'react';
import { View, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity, Dimensions } from 'react-native';
import { Text, TextInput } from 'react-native-paper';
import { RouteProp } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useChatStore } from '../../stores/chatStore';
import { useAuthStore } from '../../stores/authStore';
import { useTheme, colors, Spacing } from '../../theme';
import { ChatStackParamList } from '../../navigation/MainNavigator';
import { Message } from '../../types';
import ModernCard from '../../components/ModernCard';
import {
    EnhancedMessage,
    EmojiPicker,
    OfflineQueueIndicator,
    RichMediaUpload
} from '../../components';

type ChatScreenRouteProp = RouteProp<ChatStackParamList, 'Chat'>;
type ChatScreenNavigationProp = StackNavigationProp<ChatStackParamList, 'Chat'>;

interface Props {
    route: ChatScreenRouteProp;
    navigation: ChatScreenNavigationProp;
}

const { width } = Dimensions.get('window');

export default function ChatScreen({ route, navigation }: Props) {
    const { room } = route.params;
    const {
        messages,
        sendMessage,
        startTyping,
        stopTyping,
        setCurrentRoom,
        replyingToMessage,
        setReplyingToMessage,
        editingMessage,
        setEditingMessage,
        offlineMessages,
        messageThreads,
        loadThread
    } = useChatStore();
    const { user } = useAuthStore();
    const [messageText, setMessageText] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [showEmojiPicker, setShowEmojiPicker] = useState(false);
    const [showMediaUpload, setShowMediaUpload] = useState(false);
    const flatListRef = useRef<FlatList>(null);
    const theme = useTheme();

    const roomMessages = messages[room.id] || [];

    const setCurrentRoomCallback = useCallback((roomToSet: typeof room | null) => {
        setCurrentRoom(roomToSet);
    }, [setCurrentRoom]);

    useEffect(() => {
        setCurrentRoomCallback(room);

        // Set up navigation header with room settings button
        navigation.setOptions({
            headerRight: () => (
                <TouchableOpacity
                    style={{ marginRight: 16 }}
                    onPress={() => navigation.navigate('RoomSettings', { room })}
                >
                    <Ionicons name="settings-outline" size={24} color={theme.colors.primary} />
                </TouchableOpacity>
            ),
        });

        return () => setCurrentRoomCallback(null);
    }, [room, setCurrentRoomCallback, navigation, theme.colors.primary]);

    useEffect(() => {
        // Scroll to bottom when new messages arrive
        if (roomMessages.length > 0) {
            setTimeout(() => {
                flatListRef.current?.scrollToEnd({ animated: true });
            }, 100);
        }
    }, [roomMessages.length]);

    const handleSendMessage = () => {
        if (messageText.trim()) {
            const replyToId = replyingToMessage?.id;
            sendMessage(room.id, messageText.trim(), 'text', replyToId);
            setMessageText('');
            setReplyingToMessage(null);
            handleStopTyping();
        }
    };

    const handleTextChange = (text: string) => {
        setMessageText(text);

        if (text.length > 0 && !isTyping) {
            setIsTyping(true);
            startTyping(room.id);
        } else if (text.length === 0 && isTyping) {
            handleStopTyping();
        }
    };

    const handleStopTyping = () => {
        if (isTyping) {
            setIsTyping(false);
            stopTyping(room.id);
        }
    };

    const handleAttachment = () => {
        setShowMediaUpload(true);
    };

    const handleEmojiSelect = (emoji: string) => {
        setMessageText(prev => prev + emoji);
        setShowEmojiPicker(false);
    };

    const handleMediaUpload = (url: string, type: string, metadata?: any) => {
        // Handle media upload - send as message
        console.log('Media upload:', { url, type, metadata });
        setShowMediaUpload(false);
    };

    const renderMessage = ({ item, index }: { item: Message; index: number }) => {
        return (
            <EnhancedMessage
                message={item}
                currentUser={user!}
                onReply={() => setReplyingToMessage(item)}
                onEdit={() => setEditingMessage(item)}
                onThread={async () => {
                    // Load thread and then navigate
                    try {
                        await loadThread(item.id);
                        const thread = messageThreads[item.id];
                        if (thread) {
                            navigation.navigate('ThreadView', { thread });
                        }
                    } catch (error) {
                        console.error('Failed to load thread:', error);
                    }
                }}
            />
        );
    };

    const renderInputBar = () => (
        <View style={[styles.inputContainer, { backgroundColor: theme.colors.surface }]}>
            {/* Reply indicator */}
            {replyingToMessage && (
                <View style={[styles.replyIndicator, { backgroundColor: theme.colors.surfaceVariant }]}>
                    <View style={styles.replyContent}>
                        <Text style={[styles.replyLabel, { color: theme.colors.onSurfaceVariant }]}>
                            Replying to {replyingToMessage.sender?.username || 'Unknown'}
                        </Text>
                        <Text
                            style={[styles.replyText, { color: theme.colors.onSurface }]}
                            numberOfLines={1}
                        >
                            {replyingToMessage.content}
                        </Text>
                    </View>
                    <TouchableOpacity
                        onPress={() => setReplyingToMessage(null)}
                        style={styles.replyClose}
                    >
                        <Ionicons name="close" size={20} color={theme.colors.onSurfaceVariant} />
                    </TouchableOpacity>
                </View>
            )}

            {/* Edit indicator */}
            {editingMessage && (
                <View style={[styles.editIndicator, { backgroundColor: colors.primary[100] }]}>
                    <View style={styles.replyContent}>
                        <Text style={[styles.replyLabel, { color: colors.primary[600] }]}>
                            Editing message
                        </Text>
                        <Text
                            style={[styles.replyText, { color: theme.colors.onSurface }]}
                            numberOfLines={1}
                        >
                            {editingMessage.content}
                        </Text>
                    </View>
                    <TouchableOpacity
                        onPress={() => setEditingMessage(null)}
                        style={styles.replyClose}
                    >
                        <Ionicons name="close" size={20} color={colors.primary[600]} />
                    </TouchableOpacity>
                </View>
            )}

            <ModernCard
                variant="elevated"
                padding="sm"
                borderRadius="xl"
                style={styles.inputCard}
            >
                <View style={styles.inputRow}>
                    {/* Attachment button */}
                    <TouchableOpacity
                        style={[styles.attachButton, { backgroundColor: colors.primary[100] }]}
                        onPress={handleAttachment}
                    >
                        <Ionicons name="add" size={20} color={colors.primary[600]} />
                    </TouchableOpacity>

                    {/* Text input */}
                    <TextInput
                        style={styles.textInput}
                        placeholder={editingMessage ? "Edit message..." : "Type a message..."}
                        value={messageText}
                        onChangeText={handleTextChange}
                        onBlur={handleStopTyping}
                        multiline
                        maxLength={1000}
                        placeholderTextColor={theme.colors.onSurfaceVariant}
                        underlineColor="transparent"
                        activeUnderlineColor="transparent"
                        contentStyle={styles.textInputContent}
                        mode="flat"
                    />

                    {/* Emoji button */}
                    <TouchableOpacity
                        style={[styles.emojiButton, { backgroundColor: colors.secondary[100] }]}
                        onPress={() => setShowEmojiPicker(!showEmojiPicker)}
                    >
                        <Ionicons name="happy" size={20} color={colors.secondary[600]} />
                    </TouchableOpacity>

                    {/* Send button */}
                    <TouchableOpacity
                        style={[
                            styles.sendButton,
                            {
                                backgroundColor: messageText.trim()
                                    ? colors.primary[500]
                                    : theme.colors.surfaceVariant,
                            },
                        ]}
                        onPress={handleSendMessage}
                        disabled={!messageText.trim()}
                        activeOpacity={0.7}
                    >
                        <Ionicons
                            name={editingMessage ? "checkmark" : "send"}
                            size={18}
                            color={messageText.trim() ? 'white' : theme.colors.onSurfaceVariant}
                        />
                    </TouchableOpacity>
                </View>
            </ModernCard>
        </View>
    );

    const renderOfflineIndicator = () => (
        <OfflineQueueIndicator
            onPress={() => {
                // Handle offline queue tap
                console.log('Offline queue tapped');
            }}
        />
    );

    return (
        <KeyboardAvoidingView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            keyboardVerticalOffset={Platform.OS === 'ios' ? 88 : 0}
        >
            {/* Offline queue indicator */}
            {offlineMessages.length > 0 && renderOfflineIndicator()}

            {/* Messages List */}
            <FlatList
                ref={flatListRef}
                data={roomMessages}
                renderItem={renderMessage}
                keyExtractor={(item) => item.id}
                contentContainerStyle={styles.messagesList}
                onContentSizeChange={() => {
                    flatListRef.current?.scrollToEnd({ animated: true });
                }}
                showsVerticalScrollIndicator={false}
                inverted={false}
            />

            {/* Input Bar */}
            {renderInputBar()}

            {/* Emoji Picker Modal */}
            {showEmojiPicker && (
                <EmojiPicker
                    visible={showEmojiPicker}
                    onEmojiSelect={handleEmojiSelect}
                    onClose={() => setShowEmojiPicker(false)}
                    roomId={room.id}
                />
            )}

            {/* Media Upload Modal */}
            {showMediaUpload && (
                <RichMediaUpload
                    visible={showMediaUpload}
                    onMediaSelected={handleMediaUpload}
                    onClose={() => setShowMediaUpload(false)}
                    roomId={room.id}
                />
            )}
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    messagesList: {
        paddingHorizontal: Spacing.md,
        paddingVertical: Spacing.sm,
    },
    messageContainer: {
        flexDirection: 'row',
        alignItems: 'flex-end',
    },
    ownMessageContainer: {
        justifyContent: 'flex-end',
    },
    otherMessageContainer: {
        justifyContent: 'flex-start',
    },
    avatarContainer: {
        width: 32,
        height: 32,
        borderRadius: 16,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: Spacing.sm,
        marginBottom: 2,
    },
    avatarSpacer: {
        width: 32,
        marginRight: Spacing.sm,
    },
    avatarText: {
        fontSize: 12,
        fontWeight: 'bold',
    },
    messageBubble: {
        maxWidth: width * 0.75,
        minWidth: 60,
        padding: Spacing.md,
        borderRadius: 16,
        marginBottom: 2,
    },
    ownMessageBubble: {
        alignSelf: 'flex-end',
    },
    otherMessageBubble: {
        alignSelf: 'flex-start',
    },
    senderName: {
        fontSize: 12,
        fontWeight: '600',
        marginBottom: 2,
    },
    messageText: {
        fontSize: 16,
        lineHeight: 20,
    },
    messageFooter: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginTop: 4,
    },
    timestamp: {
        fontSize: 11,
    },
    statusContainer: {
        marginLeft: Spacing.xs,
    },
    inputContainer: {
        paddingHorizontal: Spacing.md,
        paddingVertical: Spacing.sm,
        borderTopWidth: 1,
        borderTopColor: 'rgba(0,0,0,0.1)',
    },
    inputCard: {
        backgroundColor: 'white',
    },
    inputRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        paddingHorizontal: 4,
    },
    attachButton: {
        width: 32,
        height: 32,
        borderRadius: 16,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: Spacing.sm,
        marginBottom: 4,
    },
    emojiButton: {
        width: 32,
        height: 32,
        borderRadius: 16,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: Spacing.sm,
        marginBottom: 4,
    },
    textInput: {
        flex: 1,
        maxHeight: 120,
        backgroundColor: 'transparent',
    },
    textInputContent: {
        fontSize: 16,
        paddingHorizontal: 0,
        paddingVertical: 8,
    },
    sendButton: {
        width: 32,
        height: 32,
        borderRadius: 16,
        alignItems: 'center',
        justifyContent: 'center',
        marginLeft: Spacing.sm,
        marginBottom: 4,
    },
    replyIndicator: {
        padding: Spacing.sm,
        marginBottom: Spacing.xs,
        borderRadius: 8,
        flexDirection: 'row',
        alignItems: 'center',
    },
    editIndicator: {
        padding: Spacing.sm,
        marginBottom: Spacing.xs,
        borderRadius: 8,
        flexDirection: 'row',
        alignItems: 'center',
    },
    replyContent: {
        flex: 1,
    },
    replyLabel: {
        fontSize: 12,
        fontWeight: '600',
        marginBottom: 2,
    },
    replyText: {
        fontSize: 14,
        opacity: 0.8,
    },
    replyClose: {
        padding: 4,
    },
});
