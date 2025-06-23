import React, { useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Alert,
    Modal,
    SafeAreaView,
    ActivityIndicator,
} from 'react-native';
import { apiClient } from '../services/api';

interface RichMediaUploadProps {
    visible: boolean;
    onClose: () => void;
    onMediaSelected: (url: string, type: string, metadata?: any) => void;
    roomId: string;
}

export const RichMediaUpload: React.FC<RichMediaUploadProps> = ({
    visible,
    onClose,
    onMediaSelected,
    roomId,
}) => {
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);

    const handleImageFromCamera = () => {
        // Placeholder for camera functionality
        // In a real app, you would use react-native-image-picker or expo-image-picker
        Alert.alert('Camera', 'Camera functionality would be implemented here');
    };

    const handleImageFromLibrary = () => {
        // Placeholder for image library functionality
        // In a real app, you would use react-native-image-picker or expo-image-picker
        Alert.alert('Photo Library', 'Photo library functionality would be implemented here');
    };

    const handleDocumentPicker = async () => {
        // Placeholder for document picker functionality
        // In a real app, you would use react-native-document-picker
        Alert.alert('Documents', 'Document picker functionality would be implemented here');
    };

    const uploadOptions = [
        {
            title: 'Take Photo',
            description: 'Use camera to take a photo',
            icon: '📷',
            onPress: handleImageFromCamera,
        },
        {
            title: 'Photo & Video',
            description: 'Choose from photo library',
            icon: '🖼️',
            onPress: handleImageFromLibrary,
        },
        {
            title: 'Document',
            description: 'Upload any file type',
            icon: '📄',
            onPress: handleDocumentPicker,
        },
    ];

    return (
        <Modal
            visible={visible}
            animationType="slide"
            presentationStyle="pageSheet"
            onRequestClose={onClose}
        >
            <SafeAreaView style={styles.container}>
                <View style={styles.header}>
                    <Text style={styles.title}>Upload Media</Text>
                    <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                        <Text style={styles.closeButtonText}>Cancel</Text>
                    </TouchableOpacity>
                </View>

                {isUploading ? (
                    <View style={styles.uploadingContainer}>
                        <ActivityIndicator size="large" color="#007AFF" />
                        <Text style={styles.uploadingText}>Uploading...</Text>
                        {uploadProgress > 0 && (
                            <View style={styles.progressContainer}>
                                <View style={styles.progressBar}>
                                    <View
                                        style={[
                                            styles.progressFill,
                                            { width: `${uploadProgress}%` }
                                        ]}
                                    />
                                </View>
                                <Text style={styles.progressText}>{Math.round(uploadProgress)}%</Text>
                            </View>
                        )}
                    </View>
                ) : (
                    <View style={styles.optionsContainer}>
                        {uploadOptions.map((option, index) => (
                            <TouchableOpacity
                                key={index}
                                style={styles.optionButton}
                                onPress={option.onPress}
                            >
                                <Text style={styles.optionIcon}>{option.icon}</Text>
                                <View style={styles.optionTextContainer}>
                                    <Text style={styles.optionTitle}>{option.title}</Text>
                                    <Text style={styles.optionDescription}>{option.description}</Text>
                                </View>
                                <Text style={styles.optionArrow}>→</Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                )}

                <View style={styles.footer}>
                    <Text style={styles.footerText}>
                        Supported formats: Images, Videos, Audio, Documents
                    </Text>
                    <Text style={styles.footerText}>
                        Maximum file size: 50MB
                    </Text>
                </View>
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
    uploadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 32,
    },
    uploadingText: {
        fontSize: 16,
        color: '#333',
        marginTop: 16,
        marginBottom: 24,
    },
    progressContainer: {
        width: '100%',
        alignItems: 'center',
    },
    progressBar: {
        width: '100%',
        height: 4,
        backgroundColor: '#e0e0e0',
        borderRadius: 2,
        overflow: 'hidden',
    },
    progressFill: {
        height: '100%',
        backgroundColor: '#007AFF',
    },
    progressText: {
        fontSize: 14,
        color: '#666',
        marginTop: 8,
    },
    optionsContainer: {
        flex: 1,
        padding: 16,
    },
    optionButton: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        backgroundColor: '#f8f9fa',
        borderRadius: 12,
        marginBottom: 12,
    },
    optionIcon: {
        fontSize: 24,
        marginRight: 16,
    },
    optionTextContainer: {
        flex: 1,
    },
    optionTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#333',
        marginBottom: 4,
    },
    optionDescription: {
        fontSize: 14,
        color: '#666',
    },
    optionArrow: {
        fontSize: 18,
        color: '#ccc',
    },
    footer: {
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: '#e0e0e0',
    },
    footerText: {
        fontSize: 12,
        color: '#666',
        textAlign: 'center',
        marginBottom: 4,
    },
});
