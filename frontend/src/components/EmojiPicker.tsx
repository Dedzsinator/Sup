import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  FlatList,
  TextInput,
  StyleSheet,
  Modal,
  SafeAreaView,
} from 'react-native';
import { CustomEmoji } from '../types';
import { useChatStore } from '../stores/chatStore';

interface EmojiPickerProps {
  visible: boolean;
  onClose: () => void;
  onEmojiSelect: (emoji: string) => void;
  roomId?: string;
}

// Standard emoji categories
const STANDARD_EMOJIS = [
  { category: 'Smileys', emojis: ['😀', '😂', '🤣', '😊', '😍', '🥰', '😘', '😗', '😙', '😚'] },
  { category: 'Gestures', emojis: ['👍', '👎', '👏', '🙌', '👋', '🤝', '🙏', '✋', '🤚', '👌'] },
  { category: 'Hearts', emojis: ['❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💕'] },
  { category: 'Objects', emojis: ['🎉', '🎊', '🔥', '💯', '⭐', '✨', '🎈', '🎁', '🏆', '📱'] },
];

export const EmojiPicker: React.FC<EmojiPickerProps> = ({
  visible,
  onClose,
  onEmojiSelect,
  roomId,
}) => {
  const { roomEmojis, globalEmojis, loadRoomEmojis, loadGlobalEmojis } = useChatStore();
  const [selectedCategory, setSelectedCategory] = useState('Smileys');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    if (visible) {
      loadGlobalEmojis();
      if (roomId) {
        loadRoomEmojis(roomId);
      }
    }
  }, [visible, roomId, loadGlobalEmojis, loadRoomEmojis]);

  const handleEmojiPress = (emoji: string) => {
    onEmojiSelect(emoji);
    onClose();
  };

  const renderStandardEmoji = ({ item }: { item: string }) => (
    <TouchableOpacity
      style={styles.emojiButton}
      onPress={() => handleEmojiPress(item)}
    >
      <Text style={styles.emojiText}>{item}</Text>
    </TouchableOpacity>
  );

  const renderCustomEmoji = ({ item }: { item: CustomEmoji }) => (
    <TouchableOpacity
      style={styles.emojiButton}
      onPress={() => handleEmojiPress(`:${item.name}:`)}
    >
      <Text style={styles.emojiText}>:{item.name}:</Text>
    </TouchableOpacity>
  );

  const getEmojiData = (): (string | CustomEmoji)[] => {
    if (selectedCategory === 'Custom') {
      const customEmojis = roomId ? (roomEmojis[roomId] || []) : [];
      const allCustomEmojis = [...customEmojis, ...globalEmojis];
      
      if (searchQuery) {
        return allCustomEmojis.filter(emoji =>
          emoji.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          emoji.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
        );
      }
      return allCustomEmojis;
    }

    const categoryData = STANDARD_EMOJIS.find(cat => cat.category === selectedCategory);
    return categoryData ? categoryData.emojis : [];
  };

  const renderItem = ({ item }: { item: string | CustomEmoji }) => {
    if (selectedCategory === 'Custom') {
      return renderCustomEmoji({ item: item as CustomEmoji });
    }
    return renderStandardEmoji({ item: item as string });
  };

  const categories = [...STANDARD_EMOJIS.map(cat => cat.category), 'Custom'];

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Select Emoji</Text>
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Text style={styles.closeButtonText}>Done</Text>
          </TouchableOpacity>
        </View>

        <TextInput
          style={styles.searchInput}
          placeholder="Search emojis..."
          value={searchQuery}
          onChangeText={setSearchQuery}
        />

        <View style={styles.categoryTabs}>
          {categories.map((category) => (
            <TouchableOpacity
              key={category}
              style={[
                styles.categoryTab,
                selectedCategory === category && styles.categoryTabActive,
              ]}
              onPress={() => setSelectedCategory(category)}
            >
              <Text
                style={[
                  styles.categoryTabText,
                  selectedCategory === category && styles.categoryTabTextActive,
                ]}
              >
                {category}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <FlatList
          data={getEmojiData()}
          renderItem={renderItem}
          keyExtractor={(item, index) => 
            selectedCategory === 'Custom' 
              ? (item as CustomEmoji).id 
              : `${item as string}-${index}`
          }
          numColumns={6}
          style={styles.emojiGrid}
          contentContainerStyle={styles.emojiGridContent}
        />
      </SafeAreaView>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  closeButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  closeButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  searchInput: {
    margin: 16,
    padding: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    fontSize: 16,
  },
  categoryTabs: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    marginBottom: 16,
  },
  categoryTab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 16,
  },
  categoryTabActive: {
    backgroundColor: '#007AFF',
  },
  categoryTabText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  categoryTabTextActive: {
    color: '#fff',
  },
  emojiGrid: {
    flex: 1,
  },
  emojiGridContent: {
    padding: 16,
  },
  emojiButton: {
    flex: 1,
    aspectRatio: 1,
    justifyContent: 'center',
    alignItems: 'center',
    margin: 4,
    backgroundColor: '#f8f8f8',
    borderRadius: 8,
  },
  emojiText: {
    fontSize: 24,
  },
});
