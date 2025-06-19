defmodule Sup.Autocomplete.Service do
  @moduledoc """
  Autocomplete service that interfaces with the Python-based intelligent autocomplete system.
  
  This module provides a bridge between the Elixir backend and the Python autocomplete system,
  handling requests for text completion, suggestions, and smart predictions.
  """

  require Logger
  alias HTTPoison

  @python_service_url Application.get_env(:sup, :autocomplete_service_url, "http://localhost:8000")
  @timeout 5000  # 5 second timeout
  @max_suggestions 10

  @doc """
  Gets autocomplete suggestions for a given text input.
  
  ## Parameters
  - text: The input text to get suggestions for
  - user_id: ID of the user for personalization (optional)
  - room_id: ID of the room for context (optional)
  - limit: Maximum number of suggestions to return (default: 5)
  
  ## Returns
  - {:ok, suggestions} on success
  - {:error, reason} on failure
  """
  def get_suggestions(text, opts \\ []) do
    user_id = Keyword.get(opts, :user_id)
    room_id = Keyword.get(opts, :room_id)
    limit = Keyword.get(opts, :limit, 5)
    
    # Validate input
    cond do
      String.length(text) == 0 ->
        {:ok, []}
      
      String.length(text) > 500 ->
        {:error, "text_too_long"}
      
      limit > @max_suggestions ->
        {:error, "limit_too_high"}
      
      true ->
        make_autocomplete_request(text, user_id, room_id, limit)
    end
  end

  @doc """
  Gets smart text completion for the given input.
  
  This uses the AI-powered text generator to predict likely continuations.
  """
  def get_completion(text, opts \\ []) do
    user_id = Keyword.get(opts, :user_id)
    room_id = Keyword.get(opts, :room_id)
    max_length = Keyword.get(opts, :max_length, 50)
    
    payload = %{
      "text" => text,
      "user_id" => user_id,
      "room_id" => room_id,
      "max_length" => max_length,
      "action" => "complete"
    }
    
    case make_http_request("/complete", payload) do
      {:ok, %{"completion" => completion}} ->
        {:ok, completion}
      
      {:ok, response} ->
        Logger.warning("Unexpected completion response: #{inspect(response)}")
        {:error, "invalid_response"}
      
      error ->
        error
    end
  end

  @doc """
  Trains the autocomplete system with new chat data.
  
  This should be called periodically to improve the system with recent messages.
  """
  def train_with_messages(messages) when is_list(messages) do
    payload = %{
      "messages" => Enum.map(messages, &format_message_for_training/1),
      "action" => "train"
    }
    
    case make_http_request("/train", payload) do
      {:ok, %{"status" => "success"}} ->
        {:ok, "training_started"}
      
      {:ok, response} ->
        Logger.warning("Unexpected training response: #{inspect(response)}")
        {:error, "invalid_response"}
      
      error ->
        error
    end
  end

  @doc """
  Gets health status of the autocomplete service.
  """
  def health_check do
    case make_http_request("/health", %{}) do
      {:ok, response} ->
        {:ok, response}
      
      error ->
        error
    end
  end

  @doc """
  Gets statistics and performance metrics from the autocomplete service.
  """
  def get_stats do
    case make_http_request("/stats", %{}) do
      {:ok, response} ->
        {:ok, response}
      
      error ->
        error
    end
  end

  # Private functions

  defp make_autocomplete_request(text, user_id, room_id, limit) do
    payload = %{
      "text" => text,
      "user_id" => user_id,
      "room_id" => room_id,
      "limit" => limit,
      "action" => "suggest"
    }
    
    case make_http_request("/suggest", payload) do
      {:ok, %{"suggestions" => suggestions}} when is_list(suggestions) ->
        {:ok, suggestions}
      
      {:ok, response} ->
        Logger.warning("Unexpected autocomplete response: #{inspect(response)}")
        {:error, "invalid_response"}
      
      error ->
        error
    end
  end

  defp make_http_request(endpoint, payload) do
    url = @python_service_url <> endpoint
    headers = [{"Content-Type", "application/json"}]
    body = Jason.encode!(payload)
    
    case HTTPoison.post(url, body, headers, timeout: @timeout) do
      {:ok, %HTTPoison.Response{status_code: 200, body: response_body}} ->
        case Jason.decode(response_body) do
          {:ok, parsed_response} ->
            {:ok, parsed_response}
          
          {:error, _} ->
            Logger.error("Failed to parse autocomplete service response: #{response_body}")
            {:error, "parse_error"}
        end
      
      {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
        Logger.error("Autocomplete service returned #{status_code}: #{body}")
        {:error, "service_error"}
      
      {:error, %HTTPoison.Error{reason: reason}} ->
        Logger.error("Failed to reach autocomplete service: #{inspect(reason)}")
        {:error, "connection_error"}
    end
  end

  defp format_message_for_training(message) do
    %{
      "text" => message.content,
      "user_id" => message.sender_id,
      "room_id" => message.room_id,
      "timestamp" => DateTime.to_iso8601(message.timestamp)
    }
  end
end
