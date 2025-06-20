import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    ScrollView,
    TouchableOpacity,
    Alert,
    StyleSheet,
    TextInput,
    Image,
    ActivityIndicator,
    Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFriendsStore } from '../../stores/friendsStore';
import { useTheme } from '../../theme';
import { User, FriendRequest } from '../../types';
import ModernCard from '../../components/ModernCard';
import ModernButton from '../../components/ModernButton';

const { width } = Dimensions.get('window');

type TabType = 'friends' | 'pending' | 'blocked' | 'add';

interface FriendItemProps {
    user: User;
    onMessage?: () => void;
    onCall?: () => void;
    onVideoCall?: () => void;
    onRemove?: () => void;
    onBlock?: () => void;
    onUnblock?: () => void;
}

interface FriendRequestItemProps {
    request: FriendRequest;
    onAccept: () => void;
    onDecline: () => void;
}

const FriendItem: React.FC<FriendItemProps> = ({
    user,
    onMessage,
    onCall,
    onVideoCall,
    onRemove,
    onBlock,
    onUnblock,
}) => {
    const theme = useTheme();
    const [showActions, setShowActions] = useState(false);

    const getStatusColor = () => {
        switch (user.activity_status) {
            case 'online': return '#10B981';
            case 'away': return '#F59E0B';
            case 'busy': return '#EF4444';
            case 'invisible': return '#6B7280';
            default: return '#6B7280';
        }
    };

    return (
        <ModernCard padding="md" className="mb-2">
            <TouchableOpacity
                style={styles.friendItem}
                onPress={() => setShowActions(!showActions)}
            >
                <View style={styles.friendInfo}>
                    <View style={styles.avatarContainer}>
                        {user.avatar_url ? (
                            <Image source={{ uri: user.avatar_url }} style={styles.avatar} />
                        ) : (
                            <View style={[styles.avatarPlaceholder, { backgroundColor: theme.colors.surfaceVariant }]}>
                                <Text style={[styles.avatarText, { color: theme.colors.onSurfaceVariant }]}>
                                    {user.display_name?.charAt(0).toUpperCase() || user.username.charAt(0).toUpperCase()}
                                </Text>
                            </View>
                        )}
                        <View style={[styles.statusIndicator, { backgroundColor: getStatusColor() }]} />
                    </View>

                    <View style={styles.userDetails}>
                        <Text style={[styles.displayName, { color: theme.colors.onSurface }]}>
                            {user.display_name || user.username}
                        </Text>
                        <Text style={[styles.username, { color: theme.colors.onSurfaceVariant }]}>
                            @{user.username}
                        </Text>
                        {user.custom_activity && (
                            <Text style={[styles.activity, { color: theme.colors.onSurfaceVariant }]}>
                                {user.custom_activity.type === 'playing' && '🎮 '}
                                {user.custom_activity.type === 'listening' && '🎵 '}
                                {user.custom_activity.type === 'watching' && '📺 '}
                                {user.custom_activity.name}
                            </Text>
                        )}
                    </View>

                    <View style={styles.quickActions}>
                        {onMessage && (
                            <TouchableOpacity style={styles.quickActionButton} onPress={onMessage}>
                                <Ionicons name="chatbubble" size={20} color={theme.colors.primary} />
                            </TouchableOpacity>
                        )}
                        {onCall && (
                            <TouchableOpacity style={styles.quickActionButton} onPress={onCall}>
                                <Ionicons name="call" size={20} color={theme.colors.primary} />
                            </TouchableOpacity>
                        )}
                        {onVideoCall && (
                            <TouchableOpacity style={styles.quickActionButton} onPress={onVideoCall}>
                                <Ionicons name="videocam" size={20} color={theme.colors.primary} />
                            </TouchableOpacity>
                        )}
                    </View>
                </View>

                {showActions && (
                    <View style={styles.actionButtons}>
                        {onRemove && (
                            <ModernButton
                                title="Remove Friend"
                                onPress={onRemove}
                                variant="ghost"
                                size="sm"
                                style={styles.actionButton}
                            />
                        )}
                        {onBlock && (
                            <ModernButton
                                title="Block"
                                onPress={onBlock}
                                variant="ghost"
                                size="sm"
                                style={[styles.actionButton, { borderColor: theme.colors.error }]}
                            />
                        )}
                        {onUnblock && (
                            <ModernButton
                                title="Unblock"
                                onPress={onUnblock}
                                variant="primary"
                                size="sm"
                                style={styles.actionButton}
                            />
                        )}
                    </View>
                )}
            </TouchableOpacity>
        </ModernCard>
    );
};

const FriendRequestItem: React.FC<FriendRequestItemProps> = ({ request, onAccept, onDecline }) => {
    const theme = useTheme();

    return (
        <ModernCard padding="md" className="mb-2">
            <View style={styles.friendRequestItem}>
                <View style={styles.friendInfo}>
                    <View style={styles.avatarContainer}>
                        {request.requester.avatar_url ? (
                            <Image source={{ uri: request.requester.avatar_url }} style={styles.avatar} />
                        ) : (
                            <View style={[styles.avatarPlaceholder, { backgroundColor: theme.colors.surfaceVariant }]}>
                                <Text style={[styles.avatarText, { color: theme.colors.onSurfaceVariant }]}>
                                    {request.requester.display_name?.charAt(0).toUpperCase() ||
                                        request.requester.username.charAt(0).toUpperCase()}
                                </Text>
                            </View>
                        )}
                    </View>

                    <View style={styles.userDetails}>
                        <Text style={[styles.displayName, { color: theme.colors.onSurface }]}>
                            {request.requester.display_name || request.requester.username}
                        </Text>
                        <Text style={[styles.username, { color: theme.colors.onSurfaceVariant }]}>
                            @{request.requester.username}
                        </Text>
                        <Text style={[styles.requestTime, { color: theme.colors.onSurfaceVariant }]}>
                            {new Date(request.created_at).toLocaleDateString()}
                        </Text>
                    </View>
                </View>

                <View style={styles.requestActions}>
                    <ModernButton
                        title="Accept"
                        onPress={onAccept}
                        variant="primary"
                        size="sm"
                        style={styles.requestButton}
                    />
                    <ModernButton
                        title="Decline"
                        onPress={onDecline}
                        variant="ghost"
                        size="sm"
                        style={styles.requestButton}
                    />
                </View>
            </View>
        </ModernCard>
    );
};

export default function FriendsScreen() {
    const [activeTab, setActiveTab] = useState<TabType>('friends');
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<User[]>([]);
    const [isSearching, setIsSearching] = useState(false);

    const {
        friends,
        friendRequests,
        blockedUsers,
        isLoading,
        loadFriends,
        loadFriendRequests,
        loadBlockedUsers,
        sendFriendRequest,
        acceptFriendRequest,
        declineFriendRequest,
        removeFriend,
        blockUser,
        unblockUser,
        searchUsers,
    } = useFriendsStore();

    const theme = useTheme();

    useEffect(() => {
        loadFriends();
        loadFriendRequests();
        loadBlockedUsers();
    }, []);

    const handleSearch = async (query: string) => {
        setSearchQuery(query);
        if (query.length > 2) {
            setIsSearching(true);
            const results = await searchUsers(query);
            setSearchResults(results);
            setIsSearching(false);
        } else {
            setSearchResults([]);
        }
    };

    const handleSendFriendRequest = async (identifier: string) => {
        const success = await sendFriendRequest(identifier);
        if (success) {
            Alert.alert('Success', 'Friend request sent!');
            setSearchQuery('');
            setSearchResults([]);
        } else {
            Alert.alert('Error', 'Failed to send friend request');
        }
    };

    const handleAcceptRequest = async (requestId: string) => {
        const success = await acceptFriendRequest(requestId);
        if (success) {
            Alert.alert('Success', 'Friend request accepted!');
        } else {
            Alert.alert('Error', 'Failed to accept friend request');
        }
    };

    const handleDeclineRequest = async (requestId: string) => {
        const success = await declineFriendRequest(requestId);
        if (success) {
            Alert.alert('Success', 'Friend request declined');
        } else {
            Alert.alert('Error', 'Failed to decline friend request');
        }
    };

    const handleRemoveFriend = async (friendId: string, friendName: string) => {
        Alert.alert(
            'Remove Friend',
            `Are you sure you want to remove ${friendName} from your friends?`,
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Remove',
                    style: 'destructive',
                    onPress: async () => {
                        const success = await removeFriend(friendId);
                        if (success) {
                            Alert.alert('Success', 'Friend removed');
                        } else {
                            Alert.alert('Error', 'Failed to remove friend');
                        }
                    },
                },
            ]
        );
    };

    const handleBlockUser = async (userId: string, userName: string) => {
        Alert.alert(
            'Block User',
            `Are you sure you want to block ${userName}? This will remove them from your friends list.`,
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Block',
                    style: 'destructive',
                    onPress: async () => {
                        const success = await blockUser(userId);
                        if (success) {
                            Alert.alert('Success', 'User blocked');
                        } else {
                            Alert.alert('Error', 'Failed to block user');
                        }
                    },
                },
            ]
        );
    };

    const handleUnblockUser = async (userId: string) => {
        const success = await unblockUser(userId);
        if (success) {
            Alert.alert('Success', 'User unblocked');
        } else {
            Alert.alert('Error', 'Failed to unblock user');
        }
    };

    const renderTabButton = (tab: TabType, label: string, icon: string) => (
        <TouchableOpacity
            style={[
                styles.tabButton,
                activeTab === tab && { backgroundColor: theme.colors.primary + '20' },
            ]}
            onPress={() => setActiveTab(tab)}
        >
            <Ionicons
                name={icon as any}
                size={20}
                color={activeTab === tab ? theme.colors.primary : theme.colors.onSurfaceVariant}
            />
            <Text
                style={[
                    styles.tabText,
                    {
                        color: activeTab === tab ? theme.colors.primary : theme.colors.onSurfaceVariant,
                    },
                ]}
            >
                {label}
            </Text>
            {tab === 'pending' && friendRequests.length > 0 && (
                <View style={[styles.badge, { backgroundColor: theme.colors.error }]}>
                    <Text style={styles.badgeText}>{friendRequests.length}</Text>
                </View>
            )}
        </TouchableOpacity>
    );

    const renderContent = () => {
        if (isLoading) {
            return (
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color={theme.colors.primary} />
                    <Text style={[styles.loadingText, { color: theme.colors.onSurfaceVariant }]}>
                        Loading...
                    </Text>
                </View>
            );
        }

        switch (activeTab) {
            case 'friends':
                return (
                    <ScrollView style={styles.content}>
                        {friends.length === 0 ? (
                            <View style={styles.emptyState}>
                                <Ionicons name="people-outline" size={64} color={theme.colors.onSurfaceVariant} />
                                <Text style={[styles.emptyStateText, { color: theme.colors.onSurfaceVariant }]}>
                                    No friends yet
                                </Text>
                                <Text style={[styles.emptyStateSubtext, { color: theme.colors.onSurfaceVariant }]}>
                                    Add friends to start chatting!
                                </Text>
                            </View>
                        ) : (
                            friends.map((friend) => (
                                <FriendItem
                                    key={friend.id}
                                    user={friend}
                                    onMessage={() => {/* Navigate to DM */ }}
                                    onCall={() => {/* Initiate voice call */ }}
                                    onVideoCall={() => {/* Initiate video call */ }}
                                    onRemove={() => handleRemoveFriend(friend.id, friend.display_name || friend.username)}
                                    onBlock={() => handleBlockUser(friend.id, friend.display_name || friend.username)}
                                />
                            ))
                        )}
                    </ScrollView>
                );

            case 'pending':
                return (
                    <ScrollView style={styles.content}>
                        {friendRequests.length === 0 ? (
                            <View style={styles.emptyState}>
                                <Ionicons name="mail-outline" size={64} color={theme.colors.onSurfaceVariant} />
                                <Text style={[styles.emptyStateText, { color: theme.colors.onSurfaceVariant }]}>
                                    No pending requests
                                </Text>
                            </View>
                        ) : (
                            friendRequests.map((request) => (
                                <FriendRequestItem
                                    key={request.id}
                                    request={request}
                                    onAccept={() => handleAcceptRequest(request.id)}
                                    onDecline={() => handleDeclineRequest(request.id)}
                                />
                            ))
                        )}
                    </ScrollView>
                );

            case 'blocked':
                return (
                    <ScrollView style={styles.content}>
                        {blockedUsers.length === 0 ? (
                            <View style={styles.emptyState}>
                                <Ionicons name="ban-outline" size={64} color={theme.colors.onSurfaceVariant} />
                                <Text style={[styles.emptyStateText, { color: theme.colors.onSurfaceVariant }]}>
                                    No blocked users
                                </Text>
                            </View>
                        ) : (
                            blockedUsers.map((user) => (
                                <FriendItem
                                    key={user.id}
                                    user={user}
                                    onUnblock={() => handleUnblockUser(user.id)}
                                />
                            ))
                        )}
                    </ScrollView>
                );

            case 'add':
                return (
                    <View style={styles.addFriendContainer}>
                        <ModernCard padding="lg">
                            <Text style={[styles.sectionTitle, { color: theme.colors.onSurface }]}>
                                Add Friend
                            </Text>
                            <Text style={[styles.sectionSubtitle, { color: theme.colors.onSurfaceVariant }]}>
                                Search by username, email, or friend code
                            </Text>

                            <View style={styles.searchContainer}>
                                <TextInput
                                    style={[styles.searchInput, {
                                        backgroundColor: theme.colors.surfaceVariant,
                                        color: theme.colors.onSurface
                                    }]}
                                    placeholder="Enter username, email, or friend code"
                                    placeholderTextColor={theme.colors.onSurfaceVariant}
                                    value={searchQuery}
                                    onChangeText={handleSearch}
                                />
                                <TouchableOpacity
                                    style={[styles.searchButton, { backgroundColor: theme.colors.primary }]}
                                    onPress={() => handleSearch(searchQuery)}
                                >
                                    <Ionicons name="search" size={20} color="white" />
                                </TouchableOpacity>
                            </View>

                            {isSearching && (
                                <View style={styles.searchingContainer}>
                                    <ActivityIndicator size="small" color={theme.colors.primary} />
                                    <Text style={[styles.searchingText, { color: theme.colors.onSurfaceVariant }]}>
                                        Searching...
                                    </Text>
                                </View>
                            )}

                            {searchResults.length > 0 && (
                                <View style={styles.searchResults}>
                                    {searchResults.map((user) => (
                                        <View key={user.id} style={styles.searchResultItem}>
                                            <View style={styles.searchResultInfo}>
                                                {user.avatar_url ? (
                                                    <Image source={{ uri: user.avatar_url }} style={styles.searchResultAvatar} />
                                                ) : (
                                                    <View style={[styles.searchResultAvatarPlaceholder, { backgroundColor: theme.colors.surfaceVariant }]}>
                                                        <Text style={[styles.avatarText, { color: theme.colors.onSurfaceVariant }]}>
                                                            {user.display_name?.charAt(0).toUpperCase() || user.username.charAt(0).toUpperCase()}
                                                        </Text>
                                                    </View>
                                                )}
                                                <View>
                                                    <Text style={[styles.searchResultName, { color: theme.colors.onSurface }]}>
                                                        {user.display_name || user.username}
                                                    </Text>
                                                    <Text style={[styles.searchResultUsername, { color: theme.colors.onSurfaceVariant }]}>
                                                        @{user.username}
                                                    </Text>
                                                </View>
                                            </View>
                                            <ModernButton
                                                title="Add Friend"
                                                onPress={() => handleSendFriendRequest(user.username)}
                                                variant="primary"
                                                size="sm"
                                            />
                                        </View>
                                    ))}
                                </View>
                            )}
                        </ModernCard>
                    </View>
                );

            default:
                return null;
        }
    };

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            {/* Header */}
            <View style={[styles.header, { backgroundColor: theme.colors.surface }]}>
                <Text style={[styles.headerTitle, { color: theme.colors.onSurface }]}>Friends</Text>
                <TouchableOpacity>
                    <Ionicons name="settings-outline" size={24} color={theme.colors.onSurface} />
                </TouchableOpacity>
            </View>

            {/* Tabs */}
            <View style={[styles.tabContainer, { backgroundColor: theme.colors.surface }]}>
                {renderTabButton('friends', 'Friends', 'people')}
                {renderTabButton('pending', 'Pending', 'mail')}
                {renderTabButton('blocked', 'Blocked', 'ban')}
                {renderTabButton('add', 'Add Friend', 'person-add')}
            </View>

            {/* Content */}
            {renderContent()}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0, 0, 0, 0.1)',
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: 'bold',
    },
    tabContainer: {
        flexDirection: 'row',
        paddingHorizontal: 8,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0, 0, 0, 0.1)',
    },
    tabButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
        paddingHorizontal: 8,
        borderRadius: 8,
        margin: 4,
        position: 'relative',
    },
    tabText: {
        marginLeft: 8,
        fontSize: 14,
        fontWeight: '500',
    },
    badge: {
        position: 'absolute',
        top: 4,
        right: 4,
        borderRadius: 10,
        minWidth: 20,
        height: 20,
        alignItems: 'center',
        justifyContent: 'center',
    },
    badgeText: {
        color: 'white',
        fontSize: 12,
        fontWeight: 'bold',
    },
    content: {
        flex: 1,
        padding: 16,
    },
    loadingContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    loadingText: {
        marginTop: 12,
        fontSize: 16,
    },
    emptyState: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 64,
    },
    emptyStateText: {
        fontSize: 18,
        fontWeight: '600',
        marginTop: 16,
    },
    emptyStateSubtext: {
        fontSize: 14,
        marginTop: 8,
        textAlign: 'center',
    },
    friendItem: {
        flex: 1,
    },
    friendInfo: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    avatarContainer: {
        position: 'relative',
        marginRight: 12,
    },
    avatar: {
        width: 48,
        height: 48,
        borderRadius: 24,
    },
    avatarPlaceholder: {
        width: 48,
        height: 48,
        borderRadius: 24,
        alignItems: 'center',
        justifyContent: 'center',
    },
    avatarText: {
        fontSize: 18,
        fontWeight: 'bold',
    },
    statusIndicator: {
        position: 'absolute',
        bottom: 0,
        right: 0,
        width: 16,
        height: 16,
        borderRadius: 8,
        borderWidth: 2,
        borderColor: 'white',
    },
    userDetails: {
        flex: 1,
    },
    displayName: {
        fontSize: 16,
        fontWeight: '600',
    },
    username: {
        fontSize: 14,
        marginTop: 2,
    },
    activity: {
        fontSize: 12,
        marginTop: 2,
    },
    quickActions: {
        flexDirection: 'row',
    },
    quickActionButton: {
        padding: 8,
        marginLeft: 4,
    },
    actionButtons: {
        flexDirection: 'row',
        marginTop: 12,
        paddingTop: 12,
        borderTopWidth: 1,
        borderTopColor: 'rgba(0, 0, 0, 0.1)',
    },
    actionButton: {
        marginRight: 8,
    },
    friendRequestItem: {
        flex: 1,
    },
    requestTime: {
        fontSize: 12,
        marginTop: 2,
    },
    requestActions: {
        flexDirection: 'row',
        marginTop: 12,
    },
    requestButton: {
        marginRight: 8,
    },
    addFriendContainer: {
        flex: 1,
        padding: 16,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 8,
    },
    sectionSubtitle: {
        fontSize: 14,
        marginBottom: 16,
    },
    searchContainer: {
        flexDirection: 'row',
        marginBottom: 16,
    },
    searchInput: {
        flex: 1,
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 8,
        fontSize: 16,
    },
    searchButton: {
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 8,
        marginLeft: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    searchingContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 16,
    },
    searchingText: {
        marginLeft: 8,
    },
    searchResults: {
        marginTop: 16,
    },
    searchResultItem: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0, 0, 0, 0.1)',
    },
    searchResultInfo: {
        flexDirection: 'row',
        alignItems: 'center',
        flex: 1,
    },
    searchResultAvatar: {
        width: 40,
        height: 40,
        borderRadius: 20,
        marginRight: 12,
    },
    searchResultAvatarPlaceholder: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 12,
    },
    searchResultName: {
        fontSize: 16,
        fontWeight: '600',
    },
    searchResultUsername: {
        fontSize: 14,
        marginTop: 2,
    },
});
