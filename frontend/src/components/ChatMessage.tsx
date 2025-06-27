import React from 'react'
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'

export interface Message {
    id: string
    text: string
    user_id: string
    username: string
    timestamp: string
    is_spam?: boolean
    spam_confidence?: number
    spam_model_type?: string
}

interface ChatMessageProps {
    message: Message
    currentUserId?: string
    onPress?: (message: Message) => void
}

const ChatMessage: React.FC<ChatMessageProps> = ({
    message,
    currentUserId,
    onPress,
}) => {
    const isOwnMessage = message.user_id === currentUserId
    const [feedbackSubmitted, setFeedbackSubmitted] = React.useState(false)

    const handleNotSpamPress = () => {
        // Submit feedback that this is not spam
        setFeedbackSubmitted(true)
        // Here you would call your spam feedback API
    }

    const formatTimestamp = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    return (
        <TouchableOpacity
            testID="message-container"
            style={[
                styles.container,
                isOwnMessage ? styles.ownMessage : styles.otherMessage,
            ]}
            onPress={() => onPress?.(message)}
        >
            <View style={styles.messageContent}>
                <Text style={styles.username}>{message.username}</Text>
                <Text style={styles.messageText}>{message.text}</Text>
                <Text testID="message-timestamp" style={styles.timestamp}>
                    {formatTimestamp(message.timestamp)}
                </Text>
            </View>

            {message.is_spam && (
                <View testID="spam-warning" style={styles.spamWarning}>
                    <Text style={styles.spamWarningText}>
                        ⚠️ Message flagged as potential spam
                    </Text>
                    {message.spam_confidence && (
                        <Text testID="spam-confidence" style={styles.spamConfidence}>
                            {message.spam_confidence > 0.8 ? 'High' : 'Medium'} confidence
                        </Text>
                    )}
                    <TouchableOpacity
                        testID="not-spam-button"
                        style={styles.notSpamButton}
                        onPress={handleNotSpamPress}
                    >
                        <Text style={styles.notSpamButtonText}>Not Spam</Text>
                    </TouchableOpacity>
                </View>
            )}

            {feedbackSubmitted && (
                <View testID="feedback-submitted" style={styles.feedbackSubmitted}>
                    <Text style={styles.feedbackText}>Thank you for your feedback!</Text>
                </View>
            )}
        </TouchableOpacity>
    )
}

const styles = StyleSheet.create({
    container: {
        padding: 12,
        marginVertical: 4,
        marginHorizontal: 16,
        borderRadius: 8,
        maxWidth: '80%',
    },
    ownMessage: {
        alignSelf: 'flex-end',
        backgroundColor: '#007AFF',
    },
    otherMessage: {
        alignSelf: 'flex-start',
        backgroundColor: '#F0F0F0',
    },
    messageContent: {
        flex: 1,
    },
    username: {
        fontSize: 12,
        fontWeight: 'bold',
        marginBottom: 4,
        color: '#666666',
    },
    messageText: {
        fontSize: 16,
        lineHeight: 20,
        color: '#000000',
    },
    timestamp: {
        fontSize: 10,
        color: '#999999',
        marginTop: 4,
    },
    spamWarning: {
        backgroundColor: '#FFF3CD',
        borderColor: '#FFEAA7',
        borderWidth: 1,
        borderRadius: 4,
        padding: 8,
        marginTop: 8,
    },
    spamWarningText: {
        fontSize: 12,
        color: '#856404',
        fontWeight: 'bold',
    },
    spamConfidence: {
        fontSize: 10,
        color: '#856404',
        marginTop: 2,
    },
    notSpamButton: {
        backgroundColor: '#28A745',
        borderRadius: 4,
        padding: 4,
        marginTop: 4,
        alignSelf: 'flex-start',
    },
    notSpamButtonText: {
        color: 'white',
        fontSize: 10,
        fontWeight: 'bold',
    },
    feedbackSubmitted: {
        backgroundColor: '#D4EDDA',
        borderColor: '#C3E6CB',
        borderWidth: 1,
        borderRadius: 4,
        padding: 4,
        marginTop: 4,
    },
    feedbackText: {
        fontSize: 10,
        color: '#155724',
    },
})

export default ChatMessage
