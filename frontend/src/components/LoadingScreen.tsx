import React, { useEffect } from 'react';
import { View, Animated, Easing } from 'react-native';
import { Text } from 'react-native-paper';
import { useTheme } from '../theme';

interface LoadingScreenProps {
    message?: string;
}

export default function LoadingScreen({ message = 'Loading...' }: LoadingScreenProps) {
    const theme = useTheme();
    const rotateValue = new Animated.Value(0);
    const scaleValue = new Animated.Value(1);
    const fadeValue = new Animated.Value(0);

    useEffect(() => {
        // Fade in animation
        Animated.timing(fadeValue, {
            toValue: 1,
            duration: 500,
            useNativeDriver: true,
        }).start();

        // Continuous rotation
        const rotateAnimation = Animated.loop(
            Animated.timing(rotateValue, {
                toValue: 1,
                duration: 2000,
                easing: Easing.linear,
                useNativeDriver: true,
            })
        );

        // Pulsing scale animation
        const scaleAnimation = Animated.loop(
            Animated.sequence([
                Animated.timing(scaleValue, {
                    toValue: 1.2,
                    duration: 1000,
                    easing: Easing.inOut(Easing.ease),
                    useNativeDriver: true,
                }),
                Animated.timing(scaleValue, {
                    toValue: 1,
                    duration: 1000,
                    easing: Easing.inOut(Easing.ease),
                    useNativeDriver: true,
                }),
            ])
        );

        rotateAnimation.start();
        scaleAnimation.start();

        return () => {
            rotateAnimation.stop();
            scaleAnimation.stop();
        };
    }, []);

    const rotate = rotateValue.interpolate({
        inputRange: [0, 1],
        outputRange: ['0deg', '360deg'],
    });

    return (
        <Animated.View
            className="flex-1 justify-center items-center bg-gradient-to-br from-primary-50 via-secondary-50 to-accent-50"
            style={{
                backgroundColor: theme.colors.background,
                opacity: fadeValue
            }}
        >
            {/* Modern loading spinner */}
            <View className="relative">
                {/* Outer rotating ring */}
                <Animated.View
                    className="w-20 h-20 rounded-full border-4 border-primary-200"
                    style={{
                        transform: [{ rotate }],
                        borderTopColor: theme.colors.primary,
                    }}
                />

                {/* Inner pulsing dot */}
                <Animated.View
                    className="absolute top-1/2 left-1/2 w-3 h-3 rounded-full bg-primary-500"
                    style={{
                        backgroundColor: theme.colors.primary,
                        transform: [
                            { translateX: -6 },
                            { translateY: -6 },
                            { scale: scaleValue }
                        ],
                    }}
                />
            </View>

            {/* Modern text with gradient effect */}
            <View className="mt-8 items-center">
                <Text
                    className="text-lg font-medium text-center tracking-wide"
                    style={{ color: theme.colors.onBackground }}
                >
                    {message}
                </Text>

                {/* Animated dots */}
                <View className="flex-row mt-2 space-x-1">
                    {[0, 1, 2].map((index) => (
                        <Animated.View
                            key={index}
                            className="w-2 h-2 rounded-full bg-primary-400"
                            style={{
                                backgroundColor: theme.colors.primary,
                                opacity: fadeValue,
                                transform: [{
                                    scale: scaleValue.interpolate({
                                        inputRange: [1, 1.2],
                                        outputRange: [0.5, 1],
                                    })
                                }]
                            }}
                        />
                    ))}
                </View>
            </View>

            {/* Background pattern */}
            <View className="absolute inset-0 opacity-5">
                <View className="flex-1 flex-row">
                    {Array.from({ length: 10 }).map((_, i) => (
                        <View key={i} className="flex-1">
                            {Array.from({ length: 20 }).map((_, j) => (
                                <View
                                    key={j}
                                    className="flex-1 border-r border-b border-primary-100"
                                    style={{
                                        borderColor: theme.colors.outline,
                                    }}
                                />
                            ))}
                        </View>
                    ))}
                </View>
            </View>
        </Animated.View>
    );
}
