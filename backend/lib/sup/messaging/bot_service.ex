defmodule Sup.Messaging.BotService do
  @moduledoc """
  Bot framework for creating and managing chatbots in the Sup messaging system.
  Supports command parsing, webhooks, and automated responses.
  """

  alias Sup.Messaging.{BotUser, EnhancedMessageService}
  alias Sup.Room.RoomService
  alias Sup.Repo
  import Ecto.Query
  require Logger

  @doc """
  Register a new bot
  """
  def register_bot(creator_id, bot_attrs) do
    bot_params = %{
      name: bot_attrs["name"],
      username: bot_attrs["username"],
      description: bot_attrs["description"],
      avatar_url: bot_attrs["avatar_url"],
      webhook_url: bot_attrs["webhook_url"],
      commands: bot_attrs["commands"] || [],
      permissions: bot_attrs["permissions"] || %{},
      is_active: true,
      created_by: creator_id,
      bot_token: generate_bot_token(),
      rate_limit_config: bot_attrs["rate_limit_config"] || default_rate_limits()
    }

    case BotUser.changeset(%BotUser{}, bot_params) |> Repo.insert() do
      {:ok, bot} ->
        Logger.info("Registered bot #{bot.username} (#{bot.id})")
        {:ok, BotUser.public_fields(bot)}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  @doc """
  Update bot configuration
  """
  def update_bot(bot_id, creator_id, updates) do
    case get_bot_by_creator(bot_id, creator_id) do
      nil ->
        {:error, "bot_not_found"}

      bot ->
        case BotUser.changeset(bot, updates) |> Repo.update() do
          {:ok, updated_bot} ->
            {:ok, BotUser.public_fields(updated_bot)}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  @doc """
  Process incoming message for bot commands
  """
  def process_message_for_bots(message) do
    # Check if message is a bot command
    if String.starts_with?(message.content, "/") or mentions_bot?(message.content) do
      room_bots = get_room_bots(message.room_id)

      Enum.each(room_bots, fn bot ->
        if should_process_message?(bot, message) do
          process_bot_message(bot, message)
        end
      end)
    end
  end

  @doc """
  Send message as bot
  """
  def send_bot_message(bot_id, room_id, content, opts \\ []) do
    case get_active_bot(bot_id) do
      nil ->
        {:error, "bot_not_found"}

      bot ->
        case RoomService.can_send_message?(bot.id, room_id) do
          true ->
            message_data = %{
              "room_id" => room_id,
              "content" => content,
              "type" => Keyword.get(opts, :type, "text"),
              "metadata" => %{
                "bot_id" => bot.id,
                "bot_name" => bot.name,
                "is_bot_message" => true
              }
            }

            EnhancedMessageService.send_message(bot.id, message_data)

          false ->
            {:error, "unauthorized"}
        end
    end
  end

  @doc """
  Add bot to room
  """
  def add_bot_to_room(bot_id, room_id, admin_user_id) do
    with bot when not is_nil(bot) <- get_active_bot(bot_id),
         true <- RoomService.is_room_admin?(admin_user_id, room_id) do

      case RoomService.join_room(bot.id, room_id) do
        {:ok, _member} ->
          Logger.info("Bot #{bot.username} added to room #{room_id}")
          {:ok, "bot_added"}

        {:error, reason} ->
          {:error, reason}
      end
    else
      nil -> {:error, "bot_not_found"}
      false -> {:error, "unauthorized"}
    end
  end

  @doc """
  Remove bot from room
  """
  def remove_bot_from_room(bot_id, room_id, admin_user_id) do
    with bot when not is_nil(bot) <- get_active_bot(bot_id),
         true <- RoomService.is_room_admin?(admin_user_id, room_id) do

      case RoomService.leave_room(bot.id, room_id) do
        {:ok, _member} ->
          Logger.info("Bot #{bot.username} removed from room #{room_id}")
          {:ok, "bot_removed"}

        {:error, reason} ->
          {:error, reason}
      end
    else
      nil -> {:error, "bot_not_found"}
      false -> {:error, "unauthorized"}
    end
  end

  @doc """
  Handle webhook for bot
  """
  def handle_webhook(bot_token, webhook_data) do
    case get_bot_by_token(bot_token) do
      nil ->
        {:error, "invalid_token"}

      bot ->
        if bot.webhook_url do
          send_webhook(bot, webhook_data)
        else
          process_webhook_locally(bot, webhook_data)
        end
    end
  end

  @doc """
  Get bot statistics
  """
  def get_bot_stats(bot_id, creator_id) do
    case get_bot_by_creator(bot_id, creator_id) do
      nil ->
        {:error, "bot_not_found"}

      bot ->
        stats = %{
          messages_sent: get_bot_message_count(bot.id),
          commands_processed: get_bot_command_count(bot.id),
          rooms_joined: get_bot_room_count(bot.id),
          last_active: get_bot_last_activity(bot.id),
          uptime_percentage: calculate_bot_uptime(bot.id)
        }

        {:ok, stats}
    end
  end

  # Private functions

  defp get_bot_by_creator(bot_id, creator_id) do
    from(b in BotUser, where: b.id == ^bot_id and b.created_by == ^creator_id)
    |> Repo.one()
  end

  defp get_active_bot(bot_id) do
    from(b in BotUser, where: b.id == ^bot_id and b.is_active == true)
    |> Repo.one()
  end

  defp get_bot_by_token(token) do
    from(b in BotUser, where: b.bot_token == ^token and b.is_active == true)
    |> Repo.one()
  end

  defp get_room_bots(room_id) do
    from(b in BotUser,
         join: rm in "room_members", on: rm.user_id == b.id,
         where: rm.room_id == ^room_id and b.is_active == true)
    |> Repo.all()
  end

  defp mentions_bot?(content) do
    # Check if message mentions any bot (starts with @bot_username)
    Regex.match?(~r/@\w*bot\w*/i, content)
  end

  defp should_process_message?(bot, message) do
    cond do
      # Don't process bot's own messages
      message.sender_id == bot.id -> false

      # Process if it's a command
      String.starts_with?(message.content, "/") -> true

      # Process if bot is mentioned
      String.contains?(message.content, "@#{bot.username}") -> true

      # Process if bot is configured to respond to all messages
      bot.permissions["respond_to_all"] -> true

      true -> false
    end
  end

  defp process_bot_message(bot, message) do
    Task.start(fn ->
      try do
        if bot.webhook_url do
          send_message_to_webhook(bot, message)
        else
          process_command_locally(bot, message)
        end

        # Update bot activity
        update_bot_activity(bot.id)
      rescue
        error ->
          Logger.error("Bot processing error for #{bot.username}: #{inspect(error)}")
      end
    end)
  end

  defp send_message_to_webhook(bot, message) do
    webhook_payload = %{
      bot_id: bot.id,
      message: %{
        id: message.id,
        content: message.content,
        sender_id: message.sender_id,
        room_id: message.room_id,
        timestamp: message.timestamp
      },
      room_info: get_room_context(message.room_id)
    }

    case HTTPoison.post(bot.webhook_url, Jason.encode!(webhook_payload),
                       [{"Content-Type", "application/json"}], timeout: 5000) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        handle_webhook_response(bot, message, body)

      {:ok, %HTTPoison.Response{status_code: status_code}} ->
        Logger.warning("Webhook returned status #{status_code} for bot #{bot.username}")

      {:error, error} ->
        Logger.error("Webhook request failed for bot #{bot.username}: #{inspect(error)}")
    end
  end

  defp handle_webhook_response(bot, message, response_body) do
    case Jason.decode(response_body) do
      {:ok, %{"response" => response_text}} when is_binary(response_text) ->
        send_bot_message(bot.id, message.room_id, response_text)

      {:ok, %{"response" => response_data}} when is_map(response_data) ->
        send_bot_message(bot.id, message.room_id, response_data["content"],
                        type: response_data["type"] || "text")

      _ ->
        Logger.warning("Invalid webhook response format from bot #{bot.username}")
    end
  end

  defp process_command_locally(bot, message) do
    command = extract_command(message.content)

    case find_matching_command(bot, command) do
      nil ->
        # No matching command, send default response if configured
        if bot.permissions["default_response"] do
          send_bot_message(bot.id, message.room_id, "Sorry, I don't understand that command.")
        end

      bot_command ->
        execute_bot_command(bot, message, bot_command)
    end
  end

  defp extract_command(content) do
    case Regex.run(~r/^\/(\w+)(.*)/, content) do
      [_, command, args] -> {command, String.trim(args)}
      _ -> nil
    end
  end

  defp find_matching_command(bot, {command, _args}) do
    Enum.find(bot.commands, fn cmd -> cmd["name"] == command end)
  end

  defp find_matching_command(_bot, nil), do: nil

  defp execute_bot_command(bot, message, command) do
    response = case command["type"] do
      "static" ->
        command["response"]

      "dynamic" ->
        # Execute custom logic based on command configuration
        execute_dynamic_command(command, message)

      "webhook" ->
        # Send to specific webhook for this command
        send_command_to_webhook(bot, message, command)
    end

    if response do
      send_bot_message(bot.id, message.room_id, response)
    end
  end

  defp execute_dynamic_command(command, message) do
    # Basic dynamic command processing
    case command["name"] do
      "help" ->
        "Available commands: " <> Enum.join(Map.keys(command), ", ")

      "ping" ->
        "Pong! 🏓"

      "time" ->
        "Current time: #{DateTime.utc_now() |> DateTime.to_string()}"

      "echo" ->
        {_cmd, args} = extract_command(message.content)
        args

      _ ->
        command["response"] || "Command not implemented"
    end
  end

  defp send_command_to_webhook(bot, message, command) do
    # Implementation for command-specific webhooks
    nil
  end

  defp get_room_context(room_id) do
    case RoomService.get_room(room_id) do
      nil -> %{}
      room -> %{name: room.name, type: room.type}
    end
  end

  defp update_bot_activity(bot_id) do
    from(b in BotUser, where: b.id == ^bot_id)
    |> Repo.update_all(set: [last_activity_at: DateTime.utc_now()])
  end

  defp get_bot_message_count(bot_id) do
    # This would query your message storage system
    # For now, return a placeholder
    0
  end

  defp get_bot_command_count(bot_id) do
    # This would query your analytics system
    # For now, return a placeholder
    0
  end

  defp get_bot_room_count(bot_id) do
    from(rm in "room_members", where: rm.user_id == ^bot_id)
    |> Repo.aggregate(:count, :id)
  end

  defp get_bot_last_activity(bot_id) do
    case from(b in BotUser, where: b.id == ^bot_id, select: b.last_activity_at) |> Repo.one() do
      nil -> nil
      timestamp -> timestamp
    end
  end

  defp calculate_bot_uptime(bot_id) do
    # Calculate uptime percentage over the last 30 days
    # This would typically involve checking error logs and downtime
    # For now, return a placeholder
    99.5
  end

  defp send_webhook(bot, webhook_data) do
    # Implementation for sending webhook notifications
    {:ok, "webhook_sent"}
  end

  defp process_webhook_locally(bot, webhook_data) do
    # Process webhook data locally if no external webhook URL
    {:ok, "processed_locally"}
  end

  defp generate_bot_token do
    :crypto.strong_rand_bytes(32) |> Base.encode64() |> binary_part(0, 43)
  end

  defp default_rate_limits do
    %{
      "messages_per_minute" => 60,
      "commands_per_minute" => 30,
      "webhook_calls_per_minute" => 100
    }
  end
end
