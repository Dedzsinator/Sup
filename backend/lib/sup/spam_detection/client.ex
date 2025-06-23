defmodule Sup.SpamDetection.Client do
  @moduledoc """
  Client for communicating with the spam detection microservice
  """

  require Logger

  @base_url Application.get_env(:sup, :spam_detection_url, "http://localhost:8080")
  @api_key Application.get_env(:sup, :spam_detection_api_key, "your-secret-api-key")
  @timeout 5000

  def check_spam(message, user_id, timestamp \\ nil) do
    timestamp = timestamp || DateTime.utc_now()

    payload = %{
      message: message,
      user_id: user_id,
      timestamp: DateTime.to_iso8601(timestamp)
    }

    headers = [
      {"Authorization", "Bearer #{@api_key}"},
      {"Content-Type", "application/json"}
    ]

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
    batch_messages =
      Enum.map(messages, fn msg_data ->
        base_msg = %{
          message: Map.get(msg_data, :message),
          user_id: Map.get(msg_data, :user_id)
        }

        case Map.get(msg_data, :timestamp) do
          nil -> base_msg
          timestamp -> Map.put(base_msg, :timestamp, DateTime.to_iso8601(timestamp))
        end
      end)

    payload = %{messages: batch_messages}

    headers = [
      {"Authorization", "Bearer #{@api_key}"},
      {"Content-Type", "application/json"}
    ]

    case HTTPoison.post("#{@base_url}/predict/batch", Jason.encode!(payload), headers,
           recv_timeout: @timeout * 2
         ) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, results} when is_list(results) ->
            Logger.debug("Batch spam check completed: #{length(results)} results")
            {:ok, results}

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

  def get_fallback_result do
    %{
      "is_spam" => false,
      "spam_probability" => 0.5,
      "confidence" => 0.1,
      "processing_time_ms" => 0.0,
      "error" => "Spam detection service unavailable"
    }
  end

  def check_spam_with_fallback(message, user_id, timestamp \\ nil) do
    case check_spam(message, user_id, timestamp) do
      {:ok, result} ->
        result

      {:error, _} ->
        get_fallback_result()
    end
  end
end
