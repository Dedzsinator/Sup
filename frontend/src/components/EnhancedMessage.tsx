import React, { useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Alert,
    ActionSheetIOS,
    Platform,
} from 'react-native';
import { Message, User } from '../types';
import { useChatStore } from '../stores/chatStore';
import { MessageReactions } from './MessageReactions';
import { EmojiPicker } from './EmojiPicker';

interface EnhancedMessageProps {
    message: Message;
    currentUser: User;
    onReply?: (message: Message) => void;
    onThread?: (message: Message) => void;
    onEdit?: (message: Message) => void;
}

export const EnhancedMessage: React.FC<EnhancedMessageProps> = ({
    message,
    currentUser,
    onReply,
    onThread,
    onEdit,
}) => {
    const {
        messageReactions,
        addReaction,
        deleteMessage,
    } = useChatStore();

    const [showEmojiPicker, setShowEmojiPicker] = useState(false);
    const reactions = messageReactions[message.id] || [];
    const isOwnMessage = message.sender_id === currentUser.id;

    const handleLongPress = () => {
        const options = [
            'React',
            'Reply',
            'Start Thread',
            ...(isOwnMessage ? ['Edit', 'Delete'] : []),
            'Cancel',
        ];

        const actions = {
            React: () => setShowEmojiPicker(true),
            Reply: () => onReply?.(message),
            'Start Thread': () => onThread?.(message),
            Edit: () => onEdit?.(message),
            Delete: () => handleDelete(),
        };

        if (Platform.OS === 'ios') {
            ActionSheetIOS.showActionSheetWithOptions(
                {
                    options,
                    cancelButtonIndex: options.length - 1,
                    destructiveButtonIndex: isOwnMessage ? options.length - 2 : undefined,
                },
                (buttonIndex) => {
                    const action = options[buttonIndex];
                    if (action && action !== 'Cancel') {
                        actions[action as keyof typeof actions]?.();
                    }
                }
            );
        } else {
            // For Android, you might want to use a custom action sheet or Alert
            Alert.alert(
                'Message Actions',
                'Choose an action',
                [
                    ...options.slice(0, -1).map(option => ({
                        text: option,
                        onPress: () => actions[option as keyof typeof actions]?.(),
                        style: (option === 'Delete' ? 'destructive' : 'default') as 'default' | 'destructive',
                    })),
                    { text: 'Cancel', style: 'cancel' as const },
                ]
            );
        }
    };

    const handleDelete = () => {
        Alert.alert(
            'Delete Message',
            'Are you sure you want to delete this message?',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: async () => {
                        const success = await deleteMessage(message.id);
                        if (!success) {
                            Alert.alert('Error', 'Failed to delete message');
                        }
                    },
                },
            ]
        );
    };

    const handleEmojiSelect = async (emoji: string) => {
        await addReaction(message.id, emoji);
        setShowEmojiPicker(false);
    };

    const formatTimestamp = (timestamp: string) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'now';
        if (minutes < 60) return `${minutes}m`;
        if (hours < 24) return `${hours}h`;
        if (days < 7) return `${days}d`;
        return date.toLocaleDateString();
    };

    const getStatusIndicator = () => {
        if (!isOwnMessage) return null;

        switch (message.delivery_status) {
            case 'sent':
                return <Text style={styles.statusIndicator}>✓</Text>;
            case 'delivered':
                return <Text style={styles.statusIndicator}>✓✓</Text>;
            case 'read':
                return <Text style={[styles.statusIndicator, styles.readStatus]}>✓✓</Text>;
            default:
                return null;
        }
    };

    return (
        <View style={[
            styles.container,
            isOwnMessage ? styles.ownMessage : styles.otherMessage,
        ]}>
            <TouchableOpacity
                style={styles.messageContent}
                onLongPress={handleLongPress}
                delayLongPress={500}
            >
                {/* Reply indicator */}
                {message.reply_info && (
                    <View style={styles.replyInfo}>
                        <Text style={styles.replyText} numberOfLines={1}>
                            Replying to: {message.reply_info.reply_to_content}
                        </Text>
                    </View>
                )}

                {/* Thread indicator */}
                {message.thread_info && (
                    <View style={styles.threadInfo}>
                        <Text style={styles.threadText}>
                            In thread: {message.thread_info.thread_title}
                        </Text>
                    </View>
                )}

                {/* Sender name (for other messages) */}
                {!isOwnMessage && (
                    <Text style={styles.senderName}>
                        {message.sender?.display_name || message.sender?.username || 'Unknown'}
                    </Text>
                )}

                {/* Message content */}
                <Text style={[
                    styles.messageText,
                    isOwnMessage ? styles.ownMessageText : styles.otherMessageText,
                ]}>
                    {message.content}
                </Text>

                {/* Edited indicator */}
                {message.edited_at && (
                    <Text style={styles.editedIndicator}>(edited)</Text>
                )}

                {/* Message footer */}
                <View style={styles.messageFooter}>
                    <Text style={styles.timestamp}>
                        {formatTimestamp(message.timestamp)}
                    </Text>
                    {getStatusIndicator()}
                </View>
            </TouchableOpacity>

            {/* Message reactions */}
            <MessageReactions
                messageId={message.id}
                reactions={reactions}
                currentUserId={currentUser.id}
            />

            {/* Emoji picker modal */}
            <EmojiPicker
                visible={showEmojiPicker}
                onClose={() => setShowEmojiPicker(false)}
                onEmojiSelect={handleEmojiSelect}
                roomId={message.room_id}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginVertical: 4,
        marginHorizontal: 16,
    },
    ownMessage: {
        alignItems: 'flex-end',
    },
    otherMessage: {
        alignItems: 'flex-start',
    },
    messageContent: {
        maxWidth: '80%',
        backgroundColor: '#f0f0f0',
        borderRadius: 16,
        padding: 12,
    },
    ownMessageContent: {
        backgroundColor: '#007AFF',
    },
    replyInfo: {
        backgroundColor: '#e0e0e0',
        padding: 8,
        borderRadius: 8,
        marginBottom: 8,
        borderLeftWidth: 3,
        borderLeftColor: '#007AFF',
    },
    replyText: {
        fontSize: 12,
        color: '#666',
        fontStyle: 'italic',
    },
    threadInfo: {
        backgroundColor: '#fff3cd',
        padding: 6,
        borderRadius: 6,
        marginBottom: 8,
    },
    threadText: {
        fontSize: 11,
        color: '#856404',
        fontWeight: '500',
    },
    senderName: {
        fontSize: 12,
        fontWeight: '600',
        color: '#007AFF',
        marginBottom: 4,
    },
    messageText: {
        fontSize: 16,
        lineHeight: 20,
    },
    ownMessageText: {
        color: '#fff',
    },
    otherMessageText: {
        color: '#333',
    },
    editedIndicator: {
        fontSize: 11,
        color: '#999',
        fontStyle: 'italic',
        marginTop: 4,
    },
    messageFooter: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: 8,
    },
    timestamp: {
        fontSize: 11,
        color: '#999',
    },
    statusIndicator: {
        fontSize: 12,
        color: '#999',
    },
    readStatus: {
        color: '#007AFF',
    },
});
