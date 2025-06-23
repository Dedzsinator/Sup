import React, { useState, useEffect } from 'react';
import { View, FlatList, StyleSheet } from 'react-native';
import { TextInput, Text } from 'react-native-paper';
import { StackScreenProps } from '@react-navigation/stack';
import { useChatStore } from '../../stores/chatStore';
import { useTheme, Spacing } from '../../theme';
import { ChatStackParamList } from '../../navigation/MainNavigator';
import { Message } from '../../types';
import { EnhancedMessage } from '../../components';
import { useAuthStore } from '../../stores/authStore';

type Props = StackScreenProps<ChatStackParamList, 'MessageSearch'>;

export default function MessageSearchScreen({ route, navigation }: Props) {
    const { room } = route.params;
    const { user } = useAuthStore();
    const { searchMessages, searchResults, searchQuery, setSearchQuery } = useChatStore();
    const [localQuery, setLocalQuery] = useState(searchQuery);
    const theme = useTheme();

    useEffect(() => {
        // Set up navigation title
        navigation.setOptions({
            title: room ? `Search in ${room.name}` : 'Search Messages',
        });
    }, [navigation, room]);

    const handleSearch = async (query: string) => {
        setLocalQuery(query);
        setSearchQuery(query);

        if (query.trim().length > 2) {
            await searchMessages(query, room?.id);
        }
    };

    const renderMessage = ({ item }: { item: Message }) => (
        <EnhancedMessage
            message={item}
            currentUser={user!}
            onReply={() => { }}
            onEdit={() => { }}
            onThread={() => { }}
        />
    );

    return (
        <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
            <View style={[styles.searchContainer, { backgroundColor: theme.colors.surface }]}>
                <TextInput
                    style={styles.searchInput}
                    placeholder="Search messages..."
                    value={localQuery}
                    onChangeText={handleSearch}
                    mode="outlined"
                    left={<TextInput.Icon icon="magnify" />}
                    right={
                        localQuery.length > 0 ? (
                            <TextInput.Icon
                                icon="close"
                                onPress={() => handleSearch('')}
                            />
                        ) : undefined
                    }
                />
            </View>

            <FlatList
                data={searchResults}
                renderItem={renderMessage}
                keyExtractor={(item) => item.id}
                contentContainerStyle={styles.messagesList}
                showsVerticalScrollIndicator={false}
                ListEmptyComponent={
                    <View style={styles.emptyContainer}>
                        <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
                            {localQuery.length > 0 ? 'No messages found' : 'Start typing to search messages'}
                        </Text>
                    </View>
                }
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    searchContainer: {
        padding: Spacing.md,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(0,0,0,0.1)',
    },
    searchInput: {
        backgroundColor: 'transparent',
    },
    messagesList: {
        paddingHorizontal: Spacing.md,
        paddingVertical: Spacing.sm,
    },
    emptyContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingTop: Spacing.xl * 2,
    },
    emptyText: {
        fontSize: 16,
        textAlign: 'center',
    },
});
