import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { MessageReaction } from '../types';
import { useChatStore } from '../stores/chatStore';

interface MessageReactionsProps {
  messageId: string;
  reactions: MessageReaction[];
  currentUserId: string;
}

export const MessageReactions: React.FC<MessageReactionsProps> = ({
  messageId,
  reactions,
  currentUserId,
}) => {
  const { addReaction, removeReaction } = useChatStore();

  const handleReactionPress = async (emoji: string) => {
    const userReacted = reactions.find(r => r.emoji === emoji)?.users.includes(currentUserId);
    
    if (userReacted) {
      await removeReaction(messageId, emoji);
    } else {
      await addReaction(messageId, emoji);
    }
  };

  if (!reactions || reactions.length === 0) {
    return null;
  }

  return (
    <View style={styles.container}>
      {reactions.map((reaction) => {
        const userReacted = reaction.users.includes(currentUserId);
        return (
          <TouchableOpacity
            key={reaction.emoji}
            style={[
              styles.reactionBubble,
              userReacted && styles.reactionBubbleActive,
            ]}
            onPress={() => handleReactionPress(reaction.emoji)}
          >
            <Text style={styles.emoji}>{reaction.emoji}</Text>
            <Text style={[
              styles.count,
              userReacted && styles.countActive,
            ]}>
              {reaction.count}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 4,
    gap: 4,
  },
  reactionBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  reactionBubbleActive: {
    backgroundColor: '#e3f2fd',
    borderColor: '#2196f3',
  },
  emoji: {
    fontSize: 14,
    marginRight: 4,
  },
  count: {
    fontSize: 12,
    color: '#666',
    fontWeight: '500',
  },
  countActive: {
    color: '#2196f3',
    fontWeight: '600',
  },
});
