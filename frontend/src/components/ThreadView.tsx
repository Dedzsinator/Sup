import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    FlatList,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    SafeAreaView,
    KeyboardAvoidingView,
    Platform,
} from 'react-native';
import { Message, MessageThread } from '../types';
import { useChatStore } from '../stores/chatStore';

interface ThreadViewProps {
    thread: MessageThread;
    onClose: () => void;
}

export const ThreadView: React.FC<ThreadViewProps> = ({ thread, onClose }) => {
    const {
        threadMessages,
        loadThread,
        replyToThread
    } = useChatStore();

    const [replyText, setReplyText] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const messages = threadMessages[thread.id] || [];

    useEffect(() => {
        loadThread(thread.parent_message_id);
    }, [thread.parent_message_id, loadThread]);

    const handleSendReply = async () => {
        if (!replyText.trim()) return;

        setIsLoading(true);
        try {
            const success = await replyToThread(thread.id, replyText.trim());
            if (success) {
                setReplyText('');
            }
        } catch (error) {
            console.error('Failed to send thread reply:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const renderMessage = ({ item }: { item: Message }) => (
        <View style={styles.messageContainer}>
            <View style={styles.messageHeader}>
                <Text style={styles.senderName}>
                    {item.sender?.display_name || item.sender?.username || 'Unknown'}
                </Text>
                <Text style={styles.timestamp}>
                    {new Date(item.timestamp).toLocaleTimeString()}
                </Text>
            </View>
            <Text style={styles.messageContent}>{item.content}</Text>
        </View>
    );

    return (
        <SafeAreaView style={styles.container}>
            <KeyboardAvoidingView
                style={styles.container}
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            >
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={onClose} style={styles.backButton}>
                        <Text style={styles.backButtonText}>← Back</Text>
                    </TouchableOpacity>
                    <View style={styles.headerInfo}>
                        <Text style={styles.threadTitle} numberOfLines={1}>
                            Thread: {thread.title || 'Untitled Thread'}
                        </Text>
                        <Text style={styles.threadSubtitle}>
                            {messages.length} {messages.length === 1 ? 'reply' : 'replies'}
                        </Text>
                    </View>
                </View>

                {/* Original Message */}
                <View style={styles.originalMessage}>
                    <Text style={styles.originalMessageLabel}>Original Message</Text>
                    <View style={styles.originalMessageContent}>
                        <Text style={styles.originalMessageText}>
                            {thread.title || 'Thread discussion'}
                        </Text>
                    </View>
                </View>

                {/* Thread Messages */}
                <FlatList
                    data={messages}
                    renderItem={renderMessage}
                    keyExtractor={(item) => item.id}
                    style={styles.messagesList}
                    contentContainerStyle={styles.messagesListContent}
                    showsVerticalScrollIndicator={false}
                />

                {/* Reply Input */}
                <View style={styles.replyContainer}>
                    <TextInput
                        style={styles.replyInput}
                        placeholder="Reply to thread..."
                        value={replyText}
                        onChangeText={setReplyText}
                        multiline
                        maxLength={1000}
                    />
                    <TouchableOpacity
                        style={[
                            styles.sendButton,
                            (!replyText.trim() || isLoading) && styles.sendButtonDisabled,
                        ]}
                        onPress={handleSendReply}
                        disabled={!replyText.trim() || isLoading}
                    >
                        <Text style={[
                            styles.sendButtonText,
                            (!replyText.trim() || isLoading) && styles.sendButtonTextDisabled,
                        ]}>
                            {isLoading ? 'Sending...' : 'Send'}
                        </Text>
                    </TouchableOpacity>
                </View>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: '#e0e0e0',
        backgroundColor: '#f8f9fa',
    },
    backButton: {
        marginRight: 12,
    },
    backButtonText: {
        color: '#007AFF',
        fontSize: 16,
        fontWeight: '500',
    },
    headerInfo: {
        flex: 1,
    },
    threadTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#333',
        marginBottom: 2,
    },
    threadSubtitle: {
        fontSize: 12,
        color: '#666',
    },
    originalMessage: {
        padding: 16,
        backgroundColor: '#f8f9fa',
        borderBottomWidth: 1,
        borderBottomColor: '#e0e0e0',
    },
    originalMessageLabel: {
        fontSize: 12,
        fontWeight: '600',
        color: '#666',
        marginBottom: 8,
        textTransform: 'uppercase',
    },
    originalMessageContent: {
        padding: 12,
        backgroundColor: '#fff',
        borderRadius: 8,
        borderLeftWidth: 4,
        borderLeftColor: '#007AFF',
    },
    originalMessageText: {
        fontSize: 14,
        color: '#333',
        lineHeight: 20,
    },
    messagesList: {
        flex: 1,
    },
    messagesListContent: {
        padding: 16,
    },
    messageContainer: {
        marginBottom: 16,
        padding: 12,
        backgroundColor: '#f8f9fa',
        borderRadius: 8,
    },
    messageHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    senderName: {
        fontSize: 14,
        fontWeight: '600',
        color: '#333',
    },
    timestamp: {
        fontSize: 12,
        color: '#666',
    },
    messageContent: {
        fontSize: 14,
        color: '#333',
        lineHeight: 20,
    },
    replyContainer: {
        flexDirection: 'row',
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: '#e0e0e0',
        backgroundColor: '#fff',
    },
    replyInput: {
        flex: 1,
        minHeight: 40,
        maxHeight: 100,
        paddingHorizontal: 12,
        paddingVertical: 8,
        backgroundColor: '#f5f5f5',
        borderRadius: 20,
        fontSize: 16,
        marginRight: 8,
    },
    sendButton: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        backgroundColor: '#007AFF',
        borderRadius: 20,
        justifyContent: 'center',
    },
    sendButtonDisabled: {
        backgroundColor: '#ccc',
    },
    sendButtonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    sendButtonTextDisabled: {
        color: '#999',
    },
});
