defmodule Sup.SpamDetection.Client do
  @moduledoc """
  Client for communicating with the spam detection microservice
  """

  require Logger

  @base_url Application.compile_env(:sup, :spam_detection_url, "http://localhost:8082")
  @api_key Application.compile_env(:sup, :spam_detection_api_key, "your-secret-api-key")
  @timeout 5000

  def check_spam(message, user_id, timestamp \\ nil) do
    # Updated to match the current spam detection API structure
    payload = %{
      text: message,
      user_id: user_id || "anonymous",
      metadata: %{
        timestamp: timestamp || DateTime.utc_now() |> DateTime.to_iso8601()
      }
    }

    # Updated headers - API key is optional for current server
    headers =
      case @api_key do
        "your-secret-api-key" ->
          [{"Content-Type", "application/json"}]

        key when is_binary(key) ->
          [{"Authorization", "Bearer #{key}"}, {"Content-Type", "application/json"}]

        _ ->
          [{"Content-Type", "application/json"}]
      end

    case HTTPoison.post("#{@base_url}/predict", Jason.encode!(payload), headers,
           recv_timeout: @timeout
         ) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, result} ->
            Logger.debug("Spam check result: #{inspect(result)}")
            {:ok, result}

          {:error, _} ->
            Logger.error("Failed to decode spam detection response")
            {:error, :decode_error}
        end

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        Logger.error("Spam detection API error #{status_code}: #{body}")
        {:error, :api_error}

      {:error, %HTTPoison.Error{reason: :timeout}} ->
        Logger.warning("Spam detection request timed out")
        {:error, :timeout}

      {:error, error} ->
        Logger.error("Spam detection request failed: #{inspect(error)}")
        {:error, :request_failed}
    end
  end

  def check_spam_batch(messages) when is_list(messages) do
    # Updated to match current spam detection API structure
    batch_messages =
      Enum.map(messages, fn msg_data ->
        %{
          text: Map.get(msg_data, :message) || Map.get(msg_data, :text),
          user_id: Map.get(msg_data, :user_id) || "anonymous",
          metadata: %{
            timestamp:
              case Map.get(msg_data, :timestamp) do
                nil -> DateTime.utc_now() |> DateTime.to_iso8601()
                timestamp -> DateTime.to_iso8601(timestamp)
              end
          }
        }
      end)

    payload = %{messages: batch_messages}

    # Updated headers - API key is optional for current server
    headers =
      case @api_key do
        "your-secret-api-key" ->
          [{"Content-Type", "application/json"}]

        key when is_binary(key) ->
          [{"Authorization", "Bearer #{key}"}, {"Content-Type", "application/json"}]

        _ ->
          [{"Content-Type", "application/json"}]
      end

    case HTTPoison.post("#{@base_url}/predict/batch", Jason.encode!(payload), headers,
           recv_timeout: @timeout * 2
         ) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, result} ->
            # Handle the new batch response format
            predictions = Map.get(result, "predictions", [])
            Logger.debug("Batch spam check completed: #{length(predictions)} results")
            {:ok, predictions}

          {:error, _} ->
            Logger.error("Failed to decode batch spam detection response")
            {:error, :decode_error}
        end

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        Logger.error("Batch spam detection API error #{status_code}: #{body}")
        {:error, :api_error}

      {:error, error} ->
        Logger.error("Batch spam detection request failed: #{inspect(error)}")
        {:error, :request_failed}
    end
  end

  def submit_training_data(message, user_id, is_spam, timestamp \\ nil) do
    timestamp = timestamp || DateTime.utc_now()

    payload = %{
      message: message,
      user_id: user_id,
      is_spam: is_spam,
      timestamp: DateTime.to_iso8601(timestamp)
    }

    headers = [
      {"Authorization", "Bearer #{@api_key}"},
      {"Content-Type", "application/json"}
    ]

    case HTTPoison.post("#{@base_url}/train", Jason.encode!(payload), headers,
           recv_timeout: @timeout
         ) do
      {:ok, %HTTPoison.Response{status_code: 200}} ->
        Logger.debug("Training data submitted successfully")
        :ok

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        Logger.error("Training data submission failed #{status_code}: #{body}")
        {:error, :submission_failed}

      {:error, error} ->
        Logger.error("Training data submission failed: #{inspect(error)}")
        {:error, :request_failed}
    end
  end

  def get_model_stats do
    headers = [
      {"Authorization", "Bearer #{@api_key}"},
      {"Content-Type", "application/json"}
    ]

    case HTTPoison.get("#{@base_url}/stats", headers, recv_timeout: @timeout) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, stats} ->
            {:ok, stats}

          {:error, _} ->
            {:error, :decode_error}
        end

      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        Logger.error("Failed to get model stats #{status_code}: #{body}")
        {:error, :api_error}

      {:error, error} ->
        Logger.error("Failed to get model stats: #{inspect(error)}")
        {:error, :request_failed}
    end
  end

  def health_check do
    case HTTPoison.get("#{@base_url}/health", [], recv_timeout: @timeout) do
      {:ok, %HTTPoison.Response{status_code: 200}} ->
        true

      _ ->
        false
    end
  end

  def get_fallback_result(message \\ "") do
    # Basic pattern-based spam detection when service is unavailable
    spam_patterns = [
      ~r/viagra|cialis|pharmacy/i,
      ~r/(win|won|winner).*(money|cash|prize)/i,
      ~r/(click|visit).*(link|website)/i,
      ~r/(free|cheap).*(offer|deal)/i,
      ~r/(urgent|act now|limited time)/i,
      ~r/bitcoin|crypto|investment/i,
      ~r/\$\d+/,
      ~r/(loan|debt|credit)/i
    ]

    spam_score = Enum.count(spam_patterns, &Regex.match?(&1, message))

    # Check for excessive capitals
    cap_count = String.length(Regex.replace(~r/[^A-Z]/, message, ""))
    total_chars = String.length(message)
    excessive_caps = total_chars > 0 && cap_count / total_chars > 0.5

    is_spam = spam_score > 0 || excessive_caps
    confidence = min(spam_score / 3.0, 1.0)

    %{
      "is_spam" => is_spam,
      "confidence" => confidence,
      "model_type" => "fallback_rule_based",
      "processing_time_ms" => 1.0,
      "metadata" => %{
        "score" => spam_score,
        "service_status" => "unavailable"
      }
    }
  end

  def check_spam_with_fallback(message, user_id, timestamp \\ nil) do
    case check_spam(message, user_id, timestamp) do
      {:ok, result} ->
        result

      {:error, _} ->
        Logger.warning("Spam detection service unavailable, using fallback")
        get_fallback_result(message)
    end
  end
end
