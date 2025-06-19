import React, { useEffect, useState, useRef, useCallback } from 'react';
import { View, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity, Dimensions, Alert } from 'react-native';
import { Text, TextInput } from 'react-native-paper';
import { RouteProp } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { useChatStore } from '../../stores/chatStore';
import { useAuthStore } from '../../stores/authStore';
import { useTheme, colors, Spacing } from '../../theme';
import { ChatStackParamList } from '../../navigation/MainNavigator';
import { Message } from '../../types';
import ModernCard from '../../components/ModernCard';

type ChatScreenRouteProp = RouteProp<ChatStackParamList, 'Chat'>;

interface Props {
    route: ChatScreenRouteProp;
}

const { width } = Dimensions.get('window');

export default function ChatScreen({ route }: Props) {
    const { room } = route.params;
    const { messages, sendMessage, startTyping, stopTyping, setCurrentRoom } = useChatStore();
    const { user } = useAuthStore();
    const [messageText, setMessageText] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const flatListRef = useRef<FlatList>(null);
    const theme = useTheme();

    const roomMessages = messages[room.id] || [];

    const setCurrentRoomCallback = useCallback((roomToSet: typeof room | null) => {
        setCurrentRoom(roomToSet);
    }, [setCurrentRoom]);

    useEffect(() => {
        setCurrentRoomCallback(room);
        return () => setCurrentRoomCallback(null);
    }, [room, setCurrentRoomCallback]);

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
            sendMessage(room.id, messageText.trim());
            setMessageText('');
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
        Alert.alert(
            'Attachments',
            'Choose attachment type',
            [
                { text: 'Camera', onPress: () => console.log('Camera selected') },
                { text: 'Gallery', onPress: () => console.log('Gallery selected') },
                { text: 'File', onPress: () => console.log('File selected') },
                { text: 'Cancel', style: 'cancel' },
            ]
        );
    };

    const formatTimestamp = (timestamp: string): string => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    const renderMessage = ({ item, index }: { item: Message; index: number }) => {
        const isOwn = item.sender_id === user?.id;
        const prevMessage = index > 0 ? roomMessages[index - 1] : null;
        const nextMessage = index < roomMessages.length - 1 ? roomMessages[index + 1] : null;

        const showAvatar = !prevMessage || prevMessage.sender_id !== item.sender_id;
        const isLastInGroup = !nextMessage || nextMessage.sender_id !== item.sender_id;

        return (
            <View style={[
                styles.messageContainer,
                isOwn ? styles.ownMessageContainer : styles.otherMessageContainer,
                { marginBottom: isLastInGroup ? Spacing.md : Spacing.xs }
            ]}>
                {/* Avatar for other users */}
                {!isOwn && showAvatar && (
                    <View style={[styles.avatarContainer, { backgroundColor: colors.secondary[200] }]}>
                        <Text style={[styles.avatarText, { color: colors.secondary[700] }]}>
                            {item.sender?.username?.substring(0, 2).toUpperCase() || 'U'}
                        </Text>
                    </View>
                )}

                {/* Spacer when avatar is not shown */}
                {!isOwn && !showAvatar && <View style={styles.avatarSpacer} />}

                {/* Message Bubble */}
                <View style={[
                    styles.messageBubble,
                    isOwn ? styles.ownMessageBubble : styles.otherMessageBubble,
                    { backgroundColor: isOwn ? colors.primary[500] : theme.colors.surface }
                ]}>
                    {/* Sender name for group chats */}
                    {!isOwn && showAvatar && room.type !== 'direct_message' && (
                        <Text style={[styles.senderName, { color: colors.primary[600] }]}>
                            {item.sender?.username || 'Unknown'}
                        </Text>
                    )}

                    {/* Message content */}
                    <Text style={[
                        styles.messageText,
                        { color: isOwn ? 'white' : theme.colors.onSurface }
                    ]}>
                        {item.content}
                    </Text>

                    {/* Timestamp and status */}
                    <View style={styles.messageFooter}>
                        <Text style={[
                            styles.timestamp,
                            { color: isOwn ? 'rgba(255,255,255,0.7)' : theme.colors.onSurfaceVariant }
                        ]}>
                            {formatTimestamp(item.inserted_at)}
                        </Text>

                        {/* Delivery status for own messages */}
                        {isOwn && (
                            <View style={styles.statusContainer}>
                                <Ionicons
                                    name={item.delivery_status === 'read' ? 'checkmark-done' : 'checkmark'}
                                    size={16}
                                    color={item.delivery_status === 'read' ? colors.accent[300] : 'rgba(255,255,255,0.7)'}
                                />
                            </View>
                        )}
                    </View>
                </View>
            </View>
        );
    };

    const renderInputBar = () => (
        <View style={[styles.inputContainer, { backgroundColor: theme.colors.surface }]}>
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
                        placeholder="Type a message..."
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
                            name="send"
                            size={18}
                            color={messageText.trim() ? 'white' : theme.colors.onSurfaceVariant}
                        />
                    </TouchableOpacity>
                </View>
            </ModernCard>
        </View>
    );

    return (
        <KeyboardAvoidingView
            style={[styles.container, { backgroundColor: theme.colors.background }]}
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            keyboardVerticalOffset={Platform.OS === 'ios' ? 88 : 0}
        >
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
});
