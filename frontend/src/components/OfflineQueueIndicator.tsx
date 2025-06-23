import React, { useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Animated,
} from 'react-native';
import { useChatStore } from '../stores/chatStore';

interface OfflineQueueIndicatorProps {
    onPress?: () => void;
}

export const OfflineQueueIndicator: React.FC<OfflineQueueIndicatorProps> = ({ onPress }) => {
    const { offlineMessages, loadOfflineMessages, markOfflineMessagesReceived } = useChatStore();
    const fadeAnim = React.useRef(new Animated.Value(0)).current;

    useEffect(() => {
        if (offlineMessages.length > 0) {
            Animated.timing(fadeAnim, {
                toValue: 1,
                duration: 300,
                useNativeDriver: true,
            }).start();
        } else {
            Animated.timing(fadeAnim, {
                toValue: 0,
                duration: 300,
                useNativeDriver: true,
            }).start();
        }
    }, [offlineMessages.length, fadeAnim]);

    useEffect(() => {
        // Load offline messages on mount
        loadOfflineMessages();
    }, [loadOfflineMessages]);

    const handleSyncPress = async () => {
        if (offlineMessages.length > 0) {
            const messageIds = offlineMessages.map(msg => msg.id);
            await markOfflineMessagesReceived(messageIds);
        }
        onPress?.();
    };

    if (offlineMessages.length === 0) {
        return null;
    }

    return (
        <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
            <TouchableOpacity style={styles.indicator} onPress={handleSyncPress}>
                <View style={styles.iconContainer}>
                    <View style={styles.syncIcon} />
                </View>
                <View style={styles.textContainer}>
                    <Text style={styles.title}>Offline Messages</Text>
                    <Text style={styles.subtitle}>
                        {offlineMessages.length} message{offlineMessages.length !== 1 ? 's' : ''} waiting to sync
                    </Text>
                </View>
                <Text style={styles.actionText}>Sync</Text>
            </TouchableOpacity>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        top: 50,
        left: 16,
        right: 16,
        zIndex: 1000,
    },
    indicator: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FF9500',
        borderRadius: 12,
        padding: 12,
        shadowColor: '#000',
        shadowOffset: {
            width: 0,
            height: 2,
        },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
        elevation: 5,
    },
    iconContainer: {
        marginRight: 12,
    },
    syncIcon: {
        width: 20,
        height: 20,
        borderRadius: 10,
        backgroundColor: '#fff',
        opacity: 0.9,
    },
    textContainer: {
        flex: 1,
    },
    title: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
        marginBottom: 2,
    },
    subtitle: {
        color: '#fff',
        fontSize: 12,
        opacity: 0.9,
    },
    actionText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
        paddingHorizontal: 8,
    },
});
