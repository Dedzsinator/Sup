/**
 * Spam Detection Service
 * Client-side service for communicating with the spam detection backend
 */

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:4000'

export interface SpamDetectionResult {
  is_spam: boolean
  confidence: number
  model_type: string
  processing_time_ms: number
  metadata?: {
    score?: number
    service_status?: string
  }
}

export interface MessageData {
  text: string
  user_id: string
  metadata?: {
    timestamp?: string
  }
}

/**
 * Check if a single message is spam
 */
export async function checkSpam(
  text: string,
  userId: string,
  timestamp?: string
): Promise<SpamDetectionResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/spam/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        user_id: userId,
        metadata: {
          timestamp: timestamp || new Date().toISOString(),
        },
      }),
    })

    if (!response.ok) {
      throw new Error(`API responded with status: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.warn('Spam detection service unavailable, using fallback:', error)
    return getFallbackResult(text)
  }
}

/**
 * Check multiple messages for spam in batch
 */
export async function checkSpamBatch(
  messages: MessageData[]
): Promise<SpamDetectionResult[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/spam/check/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: messages.map(msg => ({
          ...msg,
          metadata: {
            timestamp: msg.metadata?.timestamp || new Date().toISOString(),
          },
        })),
      }),
    })

    if (!response.ok) {
      throw new Error(`API responded with status: ${response.status}`)
    }

    const result = await response.json()
    return result.predictions || []
  } catch (error) {
    console.warn('Batch spam detection service unavailable, using fallback:', error)
    return messages.map(msg => getFallbackResult(msg.text))
  }
}

/**
 * Submit feedback about spam detection accuracy
 */
export async function submitSpamFeedback(
  messageId: string,
  isSpam: boolean,
  userId: string
): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/api/spam/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message_id: messageId,
        is_spam: isSpam,
        user_id: userId,
      }),
    })
  } catch (error) {
    console.warn('Failed to submit spam feedback:', error)
  }
}

/**
 * Fallback spam detection using simple pattern matching
 */
function getFallbackResult(text: string = ''): SpamDetectionResult {
  const spamPatterns = [
    /viagra|cialis|pharmacy/i,
    /(win|won|winner).*(money|cash|prize)/i,
    /(click|visit).*(link|website)/i,
    /(free|cheap).*(offer|deal)/i,
    /(urgent|act now|limited time)/i,
    /bitcoin|crypto|investment/i,
    /\$\d+/,
    /(loan|debt|credit)/i,
  ]

  const spamScore = spamPatterns.filter(pattern => pattern.test(text)).length

  // Check for excessive capitals
  const capCount = (text.match(/[A-Z]/g) || []).length
  const totalChars = text.length
  const excessiveCaps = totalChars > 0 && capCount / totalChars > 0.5

  const isSpam = spamScore > 0 || excessiveCaps
  const confidence = Math.min(spamScore / 3.0, 1.0)

  return {
    is_spam: isSpam,
    confidence,
    model_type: 'fallback_rule_based',
    processing_time_ms: 1.0,
    metadata: {
      score: spamScore,
      service_status: 'unavailable',
    },
  }
}
