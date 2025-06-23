import React, { useState, useCallback } from 'react';
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Snackbar, Typography } from '@mui/material';
import { Report, Shield, Warning } from '@mui/icons-material';
import { chatStore } from '../../stores/chatStore';
import { apiClient } from '../../utils/apiClient';

interface SpamDetectionProps {
    messageId: string;
    message: string;
    userId: string;
    isSpamFlagged?: boolean;
    spamProbability?: number;
    spamConfidence?: number;
    onSpamReport?: (messageId: string, isSpam: boolean) => void;
}

export const SpamDetectionComponent: React.FC<SpamDetectionProps> = ({
    messageId,
    message,
    userId,
    isSpamFlagged = false,
    spamProbability = 0,
    spamConfidence = 0,
    onSpamReport
}) => {
    const [reportDialogOpen, setReportDialogOpen] = useState(false);
    const [reportType, setReportType] = useState<'spam' | 'not_spam' | null>(null);
    const [reporting, setReporting] = useState(false);
    const [snackbarOpen, setSnackbarOpen] = useState(false);
    const [snackbarMessage, setSnackbarMessage] = useState('');

    const handleReportSpam = useCallback(async (isSpam: boolean) => {
        setReporting(true);

        try {
            const response = await apiClient.post('/api/spam/report', {
                message,
                is_spam: isSpam
            });

            if (response.ok) {
                setSnackbarMessage(`Message reported as ${isSpam ? 'spam' : 'not spam'} successfully`);
                setSnackbarOpen(true);
                onSpamReport?.(messageId, isSpam);
            } else {
                throw new Error('Failed to report message');
            }
        } catch (error) {
            console.error('Error reporting spam:', error);
            setSnackbarMessage('Failed to report message. Please try again.');
            setSnackbarOpen(true);
        } finally {
            setReporting(false);
            setReportDialogOpen(false);
            setReportType(null);
        }
    }, [message, messageId, onSpamReport]);

    const openReportDialog = (type: 'spam' | 'not_spam') => {
        setReportType(type);
        setReportDialogOpen(true);
    };

    const handleCloseSnackbar = () => {
        setSnackbarOpen(false);
    };

    return (
        <>
            {/* Spam Warning Banner */}
            {isSpamFlagged && (
                <Alert
                    severity="warning"
                    icon={<Warning />}
                    sx={{ mb: 1, fontSize: '0.875rem' }}
                    action={
                        <Button
                            size="small"
                            onClick={() => openReportDialog('not_spam')}
                            startIcon={<Shield />}
                        >
                            Not Spam
                        </Button>
                    }
                >
                    This message was flagged as potential spam
                    {spamProbability > 0 && (
                        <Typography variant="caption" display="block">
                            Confidence: {Math.round(spamConfidence * 100)}%,
                            Probability: {Math.round(spamProbability * 100)}%
                        </Typography>
                    )}
                </Alert>
            )}

            {/* Report Spam Button */}
            {!isSpamFlagged && (
                <IconButton
                    size="small"
                    onClick={() => openReportDialog('spam')}
                    title="Report as spam"
                    sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                >
                    <Report fontSize="small" />
                </IconButton>
            )}

            {/* Report Confirmation Dialog */}
            <Dialog open={reportDialogOpen} onClose={() => setReportDialogOpen(false)}>
                <DialogTitle>
                    {reportType === 'spam' ? 'Report Spam' : 'Report Not Spam'}
                </DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="textSecondary" paragraph>
                        Message preview:
                    </Typography>
                    <Typography
                        variant="body1"
                        sx={{
                            p: 1,
                            backgroundColor: 'grey.100',
                            borderRadius: 1,
                            maxHeight: 100,
                            overflow: 'auto'
                        }}
                    >
                        {message.length > 100 ? `${message.substring(0, 100)}...` : message}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 2 }}>
                        {reportType === 'spam'
                            ? 'Are you sure you want to report this message as spam? This will help improve our spam detection system.'
                            : 'Are you sure this message is not spam? This will help improve our spam detection accuracy.'
                        }
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button
                        onClick={() => setReportDialogOpen(false)}
                        disabled={reporting}
                    >
                        Cancel
                    </Button>
                    <Button
                        onClick={() => handleReportSpam(reportType === 'spam')}
                        disabled={reporting}
                        color={reportType === 'spam' ? 'error' : 'primary'}
                        variant="contained"
                    >
                        {reporting ? 'Reporting...' : 'Confirm'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Success/Error Snackbar */}
            <Snackbar
                open={snackbarOpen}
                autoHideDuration={4000}
                onClose={handleCloseSnackbar}
                message={snackbarMessage}
            />
        </>
    );
};

export const SpamDetectionHook = () => {
    const [checkingSpam, setCheckingSpam] = useState(false);

    const checkMessageSpam = useCallback(async (message: string) => {
        if (!message.trim()) return null;

        setCheckingSpam(true);

        try {
            const response = await apiClient.post('/api/spam/check', {
                message
            });

            if (response.ok) {
                const data = await response.json();
                return data.spam_check;
            } else {
                console.error('Failed to check spam');
                return null;
            }
        } catch (error) {
            console.error('Error checking spam:', error);
            return null;
        } finally {
            setCheckingSpam(false);
        }
    }, []);

    return {
        checkMessageSpam,
        checkingSpam
    };
};

// Spam detection context for global state
interface SpamDetectionContextType {
    spamStats: any;
    isServiceHealthy: boolean;
    checkServiceHealth: () => Promise<void>;
    getSpamStats: () => Promise<void>;
}

const SpamDetectionContext = React.createContext<SpamDetectionContextType | null>(null);

export const SpamDetectionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [spamStats, setSpamStats] = useState(null);
    const [isServiceHealthy, setIsServiceHealthy] = useState(true);

    const checkServiceHealth = useCallback(async () => {
        try {
            const response = await apiClient.get('/api/spam/health');
            setIsServiceHealthy(response.ok);
        } catch (error) {
            console.error('Spam detection service health check failed:', error);
            setIsServiceHealthy(false);
        }
    }, []);

    const getSpamStats = useCallback(async () => {
        try {
            const response = await apiClient.get('/api/spam/stats');
            if (response.ok) {
                const data = await response.json();
                setSpamStats(data.stats);
            }
        } catch (error) {
            console.error('Failed to get spam stats:', error);
        }
    }, []);

    const contextValue: SpamDetectionContextType = {
        spamStats,
        isServiceHealthy,
        checkServiceHealth,
        getSpamStats
    };

    return (
        <SpamDetectionContext.Provider value={contextValue}>
            {children}
        </SpamDetectionContext.Provider>
    );
};

export const useSpamDetection = () => {
    const context = React.useContext(SpamDetectionContext);
    if (!context) {
        throw new Error('useSpamDetection must be used within a SpamDetectionProvider');
    }
    return context;
};
