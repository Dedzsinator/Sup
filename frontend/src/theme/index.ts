import { DefaultTheme, MD3DarkTheme } from 'react-native-paper';
import { useColorScheme } from 'react-native';

// Modern color palette
export const colors = {
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
  },
  secondary: {
    50: '#fdf4ff',
    100: '#fae8ff',
    200: '#f5d0fe',
    300: '#f0abfc',
    400: '#e879f9',
    500: '#d946ef',
    600: '#c026d3',
    700: '#a21caf',
    800: '#86198f',
    900: '#701a75',
  },
  accent: {
    50: '#f0fdf4',
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d',
  },
  neutral: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    300: '#cbd5e1',
    400: '#94a3b8',
    500: '#64748b',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
  },
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
  success: {
    50: '#f0fdf4',
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d',
  },
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
  },
  error: {
    50: '#fef2f2',
    100: '#fee2e2',
    200: '#fecaca',
    300: '#fca5a5',
    400: '#f87171',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
    800: '#991b1b',
    900: '#7f1d1d',
  },
  info: {
    50: '#eff6ff',
    100: '#dbeafe',
    200: '#bfdbfe',
    300: '#93c5fd',
    400: '#60a5fa',
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8',
    800: '#1e40af',
    900: '#1e3a8a',
  },
};

const lightTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: colors.primary[500],
    primaryContainer: colors.primary[50],
    secondary: colors.secondary[500],
    secondaryContainer: colors.secondary[50],
    tertiary: colors.accent[500],
    tertiaryContainer: colors.accent[50],
    surface: '#FFFFFF',
    surfaceVariant: colors.neutral[50],
    surfaceDisabled: colors.neutral[100],
    background: colors.neutral[50],
    error: colors.error[500],
    errorContainer: colors.error[100],
    onPrimary: '#FFFFFF',
    onPrimaryContainer: colors.primary[900],
    onSecondary: '#FFFFFF',
    onSecondaryContainer: colors.secondary[900],
    onTertiary: '#FFFFFF',
    onTertiaryContainer: colors.accent[900],
    onSurface: colors.neutral[900],
    onSurfaceVariant: colors.neutral[700],
    onSurfaceDisabled: colors.neutral[400],
    onBackground: colors.neutral[900],
    onError: '#FFFFFF',
    onErrorContainer: colors.error[800],
    outline: colors.neutral[300],
    outlineVariant: colors.neutral[200],
    inverseSurface: colors.neutral[800],
    inverseOnSurface: colors.neutral[100],
    inversePrimary: colors.primary[200],
    shadow: colors.neutral[900],
    scrim: colors.neutral[900],
    backdrop: 'rgba(0, 0, 0, 0.5)',
  },
  roundness: 16,
};

const darkTheme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    primary: colors.primary[400],
    primaryContainer: colors.primary[800],
    secondary: colors.secondary[400],
    secondaryContainer: colors.secondary[800],
    tertiary: colors.accent[400],
    tertiaryContainer: colors.accent[800],
    surface: colors.neutral[800],
    surfaceVariant: colors.neutral[700],
    surfaceDisabled: colors.neutral[800],
    background: colors.neutral[900],
    error: colors.error,
    errorContainer: '#7f1d1d',
    onPrimary: colors.primary[900],
    onPrimaryContainer: colors.primary[100],
    onSecondary: colors.secondary[900],
    onSecondaryContainer: colors.secondary[100],
    onTertiary: colors.accent[900],
    onTertiaryContainer: colors.accent[100],
    onSurface: colors.neutral[100],
    onSurfaceVariant: colors.neutral[300],
    onSurfaceDisabled: colors.neutral[600],
    onBackground: colors.neutral[100],
    onError: '#FFFFFF',
    onErrorContainer: '#fecaca',
    outline: colors.neutral[600],
    outlineVariant: colors.neutral[700],
    inverseSurface: colors.neutral[100],
    inverseOnSurface: colors.neutral[800],
    inversePrimary: colors.primary[600],
    shadow: '#000000',
    scrim: '#000000',
    backdrop: 'rgba(0, 0, 0, 0.7)',
  },
  roundness: 16,
};

export const useTheme = () => {
  const isDarkMode = useColorScheme() === 'dark';
  return isDarkMode ? darkTheme : lightTheme;
};

export const theme = lightTheme; // Default theme for PaperProvider

// Animation durations and easings
export const animations = {
  duration: {
    short: 200,
    medium: 300,
    long: 500,
  },
  easing: {
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },
};

// Spacing scale
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  '2xl': 48,
  '3xl': 64,
  '4xl': 96,
};

// Typography scale
export const typography = {
  sizes: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
    '4xl': 36,
    '5xl': 48,
  },
  weights: {
    light: '300' as const,
    normal: '400' as const,
    medium: '500' as const,
    semibold: '600' as const,
    bold: '700' as const,
    extrabold: '800' as const,
  },
};

export const Colors = {
  light: lightTheme.colors,
  dark: darkTheme.colors,
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const BorderRadius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  round: 9999,
};

export const Typography = {
  xs: 12,
  sm: 14,
  md: 16,
  lg: 18,
  xl: 20,
  xxl: 24,
  xxxl: 32,
};
