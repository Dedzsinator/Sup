defmodule Sup.SpamDetection.Service do
  @moduledoc """
  Service for integrating spam detection into message processing pipeline
  """

  require Logger
  alias Sup.SpamDetection.Client
  alias Sup.Messaging.MessageService

  @doc """
  Process a message through spam detection before sending
  """
  def process_message(message, user_id, room_id, timestamp \\ nil) do
    timestamp = timestamp || DateTime.utc_now()

    # Check message for spam
    spam_result = Client.check_spam_with_fallback(message, user_id, timestamp)

    # Extract spam detection results from updated API
    is_spam = Map.get(spam_result, "is_spam", false)
    confidence = Map.get(spam_result, "confidence", 0.0)
    model_type = Map.get(spam_result, "model_type", "unknown")
    processing_time = Map.get(spam_result, "processing_time_ms", 0.0)

    Logger.info(
      "Spam detection result for user #{user_id}: is_spam=#{is_spam}, confidence=#{confidence}, model=#{model_type}, time=#{processing_time}ms"
    )

    cond do
      # High confidence spam detection - block message
      is_spam && confidence > 0.8 ->
        Logger.warning("Blocking high-confidence spam message from user #{user_id}")

        {:error, :spam_detected,
         %{
           reason: "Message blocked by spam detection",
           confidence: confidence,
           model_type: model_type
         }}

      # Medium confidence spam - flag for review but allow
      is_spam && confidence > 0.5 ->
        Logger.info("Flagging medium-confidence spam message from user #{user_id}")

        # Send message but mark as flagged
        case MessageService.send_message(user_id, room_id, message, %{
               flagged_as_spam: true,
               spam_confidence: confidence,
               spam_model_type: model_type
             }) do
          {:ok, message_data} ->
            # Submit as training data (assuming it's spam for now)
            spawn(fn -> Client.submit_training_data(message, user_id, true, timestamp) end)
            {:ok, :flagged, message_data}

          error ->
            error
        end

      # Low spam probability - allow normally
      true ->
        case MessageService.send_message(user_id, room_id, message) do
          {:ok, message_data} ->
            # Submit as training data (assuming it's ham)
            if spam_probability < 0.3 do
              spawn(fn -> Client.submit_training_data(message, user_id, false, timestamp) end)
            end

            {:ok, :allowed, message_data}

          error ->
            error
        end
    end
  end

  @doc """
  Process multiple messages in batch
  """
  def process_messages_batch(messages) when is_list(messages) do
    # Prepare messages for batch processing
    batch_data =
      Enum.map(messages, fn %{message: msg, user_id: uid, timestamp: ts} ->
        %{message: msg, user_id: uid, timestamp: ts}
      end)

    case Client.check_spam_batch(batch_data) do
      {:ok, results} ->
        # Process results
        processed_messages =
          messages
          |> Enum.zip(results)
          |> Enum.map(fn {original_msg, spam_result} ->
            process_single_batch_message(original_msg, spam_result)
          end)

        {:ok, processed_messages}

      {:error, _} ->
        # Fallback to individual processing
        Logger.warning("Batch spam detection failed, falling back to individual processing")

        processed_messages =
          Enum.map(messages, fn msg ->
            case process_message(msg.message, msg.user_id, msg.room_id, msg.timestamp) do
              {:ok, status, data} ->
                %{status: status, data: data, original: msg}

              {:error, reason, details} ->
                %{status: :error, reason: reason, details: details, original: msg}
            end
          end)

        {:ok, processed_messages}
    end
  end

  defp process_single_batch_message(original_msg, spam_result) do
    is_spam = Map.get(spam_result, "is_spam", false)
    spam_probability = Map.get(spam_result, "spam_probability", 0.0)
    confidence = Map.get(spam_result, "confidence", 0.0)

    %{
      original: original_msg,
      is_spam: is_spam,
      spam_probability: spam_probability,
      confidence: confidence,
      action: determine_action(is_spam, confidence)
    }
  end

  defp determine_action(is_spam, confidence) do
    cond do
      is_spam && confidence > 0.8 -> :block
      is_spam && confidence > 0.5 -> :flag
      true -> :allow
    end
  end

  @doc """
  Report a message as spam or ham for training
  """
  def report_message(message, user_id, is_spam, timestamp \\ nil) do
    timestamp = timestamp || DateTime.utc_now()

    case Client.submit_training_data(message, user_id, is_spam, timestamp) do
      :ok ->
        Logger.info(
          "Successfully reported message as #{if is_spam, do: "spam", else: "ham"} for user #{user_id}"
        )

        :ok

      {:error, reason} ->
        Logger.error("Failed to report message as spam/ham: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Get spam detection statistics
  """
  def get_stats do
    case Client.get_model_stats() do
      {:ok, stats} ->
        {:ok, stats}

      {:error, reason} ->
        Logger.error("Failed to get spam detection stats: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Check if spam detection service is healthy
  """
  def health_check do
    Client.health_check()
  end

  @doc """
  Process a message for spam detection without sending
  Used for checking messages before they're sent
  """
  def check_message_only(message, user_id, timestamp \\ nil) do
    timestamp = timestamp || DateTime.utc_now()

    spam_result = Client.check_spam_with_fallback(message, user_id, timestamp)

    is_spam = Map.get(spam_result, "is_spam", false)
    spam_probability = Map.get(spam_result, "spam_probability", 0.0)
    confidence = Map.get(spam_result, "confidence", 0.0)

    action = determine_action(is_spam, confidence)

    %{
      is_spam: is_spam,
      spam_probability: spam_probability,
      confidence: confidence,
      action: action,
      processing_time_ms: Map.get(spam_result, "processing_time_ms", 0.0)
    }
  end
end
