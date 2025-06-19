import React, { useEffect, useState, useCallback } from 'react';
import { View, FlatList, StyleSheet, RefreshControl, TouchableOpacity } from 'react-native';
import { Text, Searchbar } from 'react-native-paper';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useChatStore } from '../../stores/chatStore';
import { useTheme, colors, Spacing } from '../../theme';
import { ChatStackParamList } from '../../navigation/MainNavigator';
import { Room } from '../../types';
import ModernCard from '../../components/ModernCard';

type ChatListScreenNavigationProp = StackNavigationProp<ChatStackParamList, 'ChatList'>;

interface Props {
    navigation: ChatListScreenNavigationProp;
}

export default function ChatListScreen({ navigation }: Props) {
    const { rooms, loadRooms } = useChatStore();
    const [searchQuery, setSearchQuery] = useState('');
    const [refreshing, setRefreshing] = useState(false);
    const theme = useTheme();

    const loadRoomsCallback = useCallback(() => {
        loadRooms();
    }, [loadRooms]);

    useEffect(() => {
        loadRoomsCallback();
    }, [loadRoomsCallback]);

    const onRefresh = async () => {
        setRefreshing(true);
        await loadRooms();
        setRefreshing(false);
    };

    const filteredRooms = rooms.filter(room =>
        room.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const renderRoom = ({ item, index }: { item: Room; index: number }) => (
        <ModernCard
            variant="elevated"
            padding="md"
            borderRadius="lg"
            style={styles.roomCard}
        >
            <TouchableOpacity
                style={styles.roomContent}
                onPress={() => navigation.navigate('Chat', { room: item })}
                activeOpacity={0.7}
            >
                {/* Room Icon */}
                <View style={[styles.iconContainer, {
                    backgroundColor: item.type === 'direct_message'
                        ? colors.primary[100]
                        : colors.secondary[100]
                }]}>
                    <Ionicons
                        name={item.type === 'direct_message' ? 'person' : 'people'}
                        size={24}
                        color={item.type === 'direct_message'
                            ? colors.primary[600]
                            : colors.secondary[600]
                        }
                    />
                </View>

                {/* Room Info */}
                <View style={styles.roomInfo}>
                    <View style={styles.roomHeader}>
                        <Text style={[styles.roomName, { color: theme.colors.onSurface }]}>
                            {item.name}
                        </Text>
                        <Text style={[styles.timestamp, { color: theme.colors.onSurfaceVariant }]}>
                            {formatTimestamp(item.updated_at)}
                        </Text>
                    </View>

                    <View style={styles.roomSubHeader}>
                        <Text
                            style={[styles.roomDescription, { color: theme.colors.onSurfaceVariant }]}
                            numberOfLines={1}
                        >
                            {item.description || 'No recent messages'}
                        </Text>

                        {/* Unread Badge */}
                        {item.unread_count && item.unread_count > 0 && (
                            <View style={[styles.unreadBadge, { backgroundColor: colors.primary[500] }]}>
                                <Text style={[styles.unreadText, { color: 'white' }]}>
                                    {item.unread_count > 99 ? '99+' : item.unread_count}
                                </Text>
                            </View>
                        )}
                    </View>
                </View>

                {/* Room Type Badge */}
                <View style={styles.roomActions}>
                    {item.type === 'direct_message' && (
                        <View style={[styles.typeBadge, { backgroundColor: colors.accent[100] }]}>
                            <Text style={[styles.typeBadgeText, { color: colors.accent[600] }]}>
                                DM
                            </Text>
                        </View>
                    )}
                    <Ionicons
                        name="chevron-forward"
                        size={20}
                        color={theme.colors.onSurfaceVariant}
                        style={styles.chevron}
                    />
                </View>
            </TouchableOpacity>
        </ModernCard>
    );

    const renderEmptyState = () => (
        <View style={styles.emptyContainer}>
            <View style={[styles.emptyIconContainer, { backgroundColor: colors.primary[100] }]}>
                <Ionicons name="chatbubbles-outline" size={48} color={colors.primary[500]} />
            </View>
            <Text style={[styles.emptyTitle, { color: theme.colors.onSurface }]}>
                No chats yet
            </Text>
            <Text style={[styles.emptySubtitle, { color: theme.colors.onSurfaceVariant }]}>
                Start a conversation or create a group chat to get started
            </Text>
        </View>
    );

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            {/* Header */}
            <View style={styles.header}>
                <Text style={[styles.headerTitle, { color: theme.colors.onBackground }]}>
                    Chats
                </Text>
            </View>

            {/* Search Bar */}
            <View style={styles.searchContainer}>
                <Searchbar
                    placeholder="Search conversations..."
                    onChangeText={setSearchQuery}
                    value={searchQuery}
                    style={[styles.searchbar, { backgroundColor: theme.colors.surfaceVariant }]}
                    inputStyle={{ color: theme.colors.onSurface }}
                    iconColor={theme.colors.onSurfaceVariant}
                    placeholderTextColor={theme.colors.onSurfaceVariant}
                    elevation={0}
                />
            </View>

            {/* Chat List */}
            <View style={styles.listContainer}>
                {filteredRooms.length === 0 ? (
                    renderEmptyState()
                ) : (
                    <FlatList
                        data={filteredRooms}
                        renderItem={renderRoom}
                        keyExtractor={(item) => item.id}
                        refreshControl={
                            <RefreshControl
                                refreshing={refreshing}
                                onRefresh={onRefresh}
                                colors={[colors.primary[500]]}
                                tintColor={colors.primary[500]}
                            />
                        }
                        contentContainerStyle={styles.listContent}
                        showsVerticalScrollIndicator={false}
                    />
                )}
            </View>

            {/* Floating Action Button */}
            <TouchableOpacity
                style={[styles.fab, {
                    backgroundColor: colors.primary[500],
                    shadowColor: colors.primary[500]
                }]}
                onPress={() => {
                    navigation.navigate('CreateRoom');
                }}
                activeOpacity={0.8}
            >
                <Ionicons name="add" size={24} color="white" />
            </TouchableOpacity>
        </View>
    );
}

const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = Math.abs(now.getTime() - date.getTime()) / 36e5;

    if (diffInHours < 24) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffInHours < 168) { // 7 days
        return date.toLocaleDateString([], { weekday: 'short' });
    } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    header: {
        paddingHorizontal: Spacing.lg,
        paddingTop: Spacing.lg,
        paddingBottom: Spacing.md,
    },
    headerTitle: {
        fontSize: 28,
        fontWeight: 'bold',
    },
    searchContainer: {
        paddingHorizontal: Spacing.lg,
        paddingBottom: Spacing.md,
    },
    searchbar: {
        borderRadius: 16,
        elevation: 0,
    },
    listContainer: {
        flex: 1,
    },
    listContent: {
        padding: Spacing.lg,
        paddingTop: Spacing.sm,
    },
    roomCard: {
        marginBottom: Spacing.sm,
    },
    roomContent: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    iconContainer: {
        width: 48,
        height: 48,
        borderRadius: 24,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: Spacing.md,
    },
    roomInfo: {
        flex: 1,
    },
    roomHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 2,
    },
    roomName: {
        fontSize: 16,
        fontWeight: '600',
        flex: 1,
    },
    timestamp: {
        fontSize: 12,
        fontWeight: '500',
    },
    roomSubHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    roomDescription: {
        fontSize: 14,
        flex: 1,
        marginRight: Spacing.sm,
    },
    unreadBadge: {
        minWidth: 20,
        height: 20,
        borderRadius: 10,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 6,
    },
    unreadText: {
        fontSize: 12,
        fontWeight: 'bold',
    },
    roomActions: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    typeBadge: {
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 12,
        marginRight: Spacing.xs,
    },
    typeBadgeText: {
        fontSize: 10,
        fontWeight: 'bold',
    },
    chevron: {
        marginLeft: Spacing.xs,
    },
    emptyContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: Spacing.xl,
    },
    emptyIconContainer: {
        width: 96,
        height: 96,
        borderRadius: 48,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.lg,
    },
    emptyTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: Spacing.sm,
    },
    emptySubtitle: {
        fontSize: 16,
        textAlign: 'center',
        lineHeight: 24,
    },
    fab: {
        position: 'absolute',
        bottom: 24,
        right: 24,
        width: 56,
        height: 56,
        borderRadius: 28,
        alignItems: 'center',
        justifyContent: 'center',
        elevation: 8,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
    },
});
