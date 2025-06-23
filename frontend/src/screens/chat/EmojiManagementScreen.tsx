import React, { useState, useEffect } from 'react';
import { View, FlatList, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Text, FAB, Card } from 'react-native-paper';
import { StackScreenProps } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useChatStore } from '../../stores/chatStore';
import { useTheme, colors, Spacing } from '../../theme';
import { ChatStackParamList } from '../../navigation/MainNavigator';
import { CustomEmoji } from '../../types';

type Props = StackScreenProps<ChatStackParamList, 'EmojiManagement'>;

export default function EmojiManagementScreen({ route, navigation }: Props) {
  const { room } = route.params;
  const { 
    roomEmojis, 
    globalEmojis, 
    loadRoomEmojis, 
    loadGlobalEmojis,
    deleteCustomEmoji
  } = useChatStore();
  const [showGlobal, setShowGlobal] = useState(false);
  const theme = useTheme();

  useEffect(() => {
    navigation.setOptions({
      title: `Emojis - ${room.name}`,
    });

    // Load emojis
    loadRoomEmojis(room.id);
    loadGlobalEmojis();
  }, [navigation, room, loadRoomEmojis, loadGlobalEmojis]);

  const currentEmojis = showGlobal ? globalEmojis : roomEmojis[room.id] || [];

  const handleDeleteEmoji = (emoji: CustomEmoji) => {
    Alert.alert(
      'Delete Emoji',
      `Are you sure you want to delete :${emoji.name}:?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => deleteCustomEmoji(room.id, emoji.id),
        },
      ]
    );
  };

  const handleCreateEmoji = () => {
    // Navigate to emoji creation screen (could be a modal)
    Alert.alert(
      'Create Emoji',
      'This feature allows you to upload custom emojis for your room.',
      [{ text: 'OK' }]
    );
  };

  const renderEmoji = ({ item }: { item: CustomEmoji }) => (
    <Card style={styles.emojiCard}>
      <Card.Content>
        <View style={styles.emojiHeader}>
          <View style={styles.emojiInfo}>
            <Text style={styles.emojiName}>:{item.name}:</Text>
            <Text style={[styles.emojiDescription, { color: theme.colors.onSurfaceVariant }]}>
              {item.description || 'No description'}
            </Text>
            <Text style={[styles.emojiUsage, { color: theme.colors.onSurfaceVariant }]}>
              Used {item.usage_count} times
            </Text>
          </View>
          <View style={styles.emojiActions}>
            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: colors.error[100] }]}
              onPress={() => handleDeleteEmoji(item)}
            >
              <Ionicons name="trash" size={16} color={colors.error[600]} />
            </TouchableOpacity>
          </View>
        </View>
        
        {item.tags.length > 0 && (
          <View style={styles.tagsContainer}>
            {item.tags.map((tag, index) => (
              <View key={index} style={[styles.tag, { backgroundColor: theme.colors.surfaceVariant }]}>
                <Text style={[styles.tagText, { color: theme.colors.onSurfaceVariant }]}>
                  {tag}
                </Text>
              </View>
            ))}
          </View>
        )}
      </Card.Content>
    </Card>
  );

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <View style={[styles.tabContainer, { backgroundColor: theme.colors.surface }]}>
        <TouchableOpacity
          style={[
            styles.tab,
            !showGlobal && { backgroundColor: colors.primary[100] }
          ]}
          onPress={() => setShowGlobal(false)}
        >
          <Text style={[
            styles.tabText,
            { color: !showGlobal ? colors.primary[600] : theme.colors.onSurface }
          ]}>
            Room Emojis ({roomEmojis[room.id]?.length || 0})
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[
            styles.tab,
            showGlobal && { backgroundColor: colors.primary[100] }
          ]}
          onPress={() => setShowGlobal(true)}
        >
          <Text style={[
            styles.tabText,
            { color: showGlobal ? colors.primary[600] : theme.colors.onSurface }
          ]}>
            Global Emojis ({globalEmojis.length})
          </Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={currentEmojis}
        renderItem={renderEmoji}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.emojiList}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Ionicons 
              name="happy-outline" 
              size={64} 
              color={theme.colors.onSurfaceVariant} 
            />
            <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
              {showGlobal ? 'No global emojis available' : 'No custom emojis in this room'}
            </Text>
          </View>
        }
      />

      {!showGlobal && (
        <FAB
          icon="plus"
          label="Add Emoji"
          style={[styles.fab, { backgroundColor: colors.primary[500] }]}
          onPress={handleCreateEmoji}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  tabContainer: {
    flexDirection: 'row',
    padding: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0,0,0,0.1)',
  },
  tab: {
    flex: 1,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    marginHorizontal: Spacing.xs,
    borderRadius: 8,
    alignItems: 'center',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
  },
  emojiList: {
    padding: Spacing.md,
  },
  emojiCard: {
    marginBottom: Spacing.sm,
  },
  emojiHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  emojiInfo: {
    flex: 1,
  },
  emojiName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  emojiDescription: {
    fontSize: 14,
    marginBottom: 2,
  },
  emojiUsage: {
    fontSize: 12,
  },
  emojiActions: {
    flexDirection: 'row',
  },
  actionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: Spacing.xs,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: Spacing.sm,
  },
  tag: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: 12,
    marginRight: Spacing.xs,
    marginBottom: Spacing.xs,
  },
  tagText: {
    fontSize: 12,
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
    marginTop: Spacing.md,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
});
