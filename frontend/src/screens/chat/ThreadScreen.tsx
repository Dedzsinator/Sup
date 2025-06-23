import React from 'react';
import { View, StyleSheet } from 'react-native';
import { StackScreenProps } from '@react-navigation/stack';
import { ThreadView } from '../../components';
import { ChatStackParamList } from '../../navigation/MainNavigator';

type Props = StackScreenProps<ChatStackParamList, 'ThreadView'>;

export default function ThreadScreen({ route, navigation }: Props) {
    const { thread } = route.params;

    const handleClose = () => {
        navigation.goBack();
    };

    return (
        <View style={styles.container}>
            <ThreadView
                thread={thread}
                onClose={handleClose}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
});
