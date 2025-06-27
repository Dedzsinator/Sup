import { checkSpam, checkSpamBatch } from '../spamDetection'

// Mock fetch
global.fetch = jest.fn()

const mockFetch = fetch as jest.MockedFunction<typeof fetch>

describe('Spam Detection Service', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('checkSpam', () => {
    it('should return spam detection result', async () => {
      const mockResponse = {
        is_spam: true,
        confidence: 0.95,
        model_type: 'neural_network',
        processing_time_ms: 25.3,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response)

      const result = await checkSpam('Buy cheap viagra now!', 'user123')

      expect(result).toEqual(mockResponse)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/spam/check'),
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: expect.stringContaining('Buy cheap viagra now!'),
        })
      )
    })

    it('should handle API errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      } as Response)

      const result = await checkSpam('test message', 'user123')

      expect(result).toEqual({
        is_spam: false,
        confidence: 0,
        model_type: 'fallback_rule_based',
        processing_time_ms: expect.any(Number),
        metadata: {
          score: 0,
          service_status: 'unavailable',
        },
      })
    })

    it('should use fallback detection when service is unavailable', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      const result = await checkSpam('WIN MONEY NOW!', 'user123')

      expect(result.is_spam).toBe(true)
      expect(result.model_type).toBe('fallback_rule_based')
      expect(result.metadata.service_status).toBe('unavailable')
    })
  })

  describe('checkSpamBatch', () => {
    it('should process multiple messages', async () => {
      const mockResponse = {
        predictions: [
          { is_spam: true, confidence: 0.9 },
          { is_spam: false, confidence: 0.1 },
        ],
        total_processed: 2,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response)

      const messages = [
        { text: 'Buy now!', user_id: 'user1' },
        { text: 'Hello world', user_id: 'user2' },
      ]

      const result = await checkSpamBatch(messages)

      expect(result).toEqual(mockResponse.predictions)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/spam/check/batch'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Buy now!'),
        })
      )
    })
  })

  describe('fallback spam detection', () => {
    it('should detect common spam patterns', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Service unavailable'))

      const testCases = [
        { message: 'Buy cheap viagra now!', expected: true },
        { message: 'WIN $1000 NOW!', expected: true },
        { message: 'CLICK HERE FOR FREE MONEY', expected: true },
        { message: 'Hello, how are you?', expected: false },
        { message: 'Meeting at 3pm today', expected: false },
      ]

      for (const testCase of testCases) {
        const result = await checkSpam(testCase.message, 'user123')
        expect(result.is_spam).toBe(testCase.expected)
      }
    })

    it('should detect excessive capitals', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Service unavailable'))

      const result = await checkSpam('THIS IS ALL CAPS!!! URGENT!!!', 'user123')
      expect(result.is_spam).toBe(true)
    })
  })
})
