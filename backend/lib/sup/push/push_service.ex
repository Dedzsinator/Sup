defmodule Sup.Push.PushService do
  @moduledoc """
  Push notification service for mobile and web notifications.
  """

  use GenServer
  require Logger

  alias Sup.Auth.User
  alias Sup.Repo

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def send_message_notification(user_id, message, room_name) do
    GenServer.cast(__MODULE__, {:send_message_notification, user_id, message, room_name})
  end

  def register_push_token(user_id, token, platform) do
    GenServer.call(__MODULE__, {:register_push_token, user_id, token, platform})
  end

  def unregister_push_token(user_id) do
    GenServer.call(__MODULE__, {:unregister_push_token, user_id})
  end

  # GenServer callbacks
  @impl true
  def init(_opts) do
    # Initialize push notification services (FCM, APNS, Web Push)
    state = %{
      fcm_key: Application.get_env(:sup, :fcm_server_key),
      apns_cert: Application.get_env(:sup, :apns_cert_path),
      web_push_keys: Application.get_env(:sup, :web_push_keys)
    }

    Logger.info("Push notification service started")
    {:ok, state}
  end

  @impl true
  def handle_cast({:send_message_notification, user_id, message, room_name}, state) do
    # Check if user should receive push notification
    case should_send_push?(user_id) do
      true ->
        case get_user_push_token(user_id) do
          {:ok, token, platform} ->
            send_push_notification(token, platform, message, room_name, state)

          {:error, _reason} ->
            Logger.debug("No push token for user #{user_id}")
        end

      false ->
        Logger.debug("User #{user_id} should not receive push notification")
    end

    {:noreply, state}
  end

  @impl true
  def handle_call({:register_push_token, user_id, token, platform}, _from, state) do
    case Repo.get(User, user_id) do
      nil ->
        {:reply, {:error, "user_not_found"}, state}

      user ->
        changeset = User.changeset(user, %{push_token: token})

        case Repo.update(changeset) do
          {:ok, _updated_user} ->
            # Store platform info in Redis for quick access
            Sup.Redis.hset("push_tokens:#{user_id}", "token", token)
            Sup.Redis.hset("push_tokens:#{user_id}", "platform", platform)

            {:reply, :ok, state}

          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
    end
  end

  def handle_call({:unregister_push_token, user_id}, _from, state) do
    case Repo.get(User, user_id) do
      nil ->
        {:reply, {:error, "user_not_found"}, state}

      user ->
        changeset = User.changeset(user, %{push_token: nil})

        case Repo.update(changeset) do
          {:ok, _updated_user} ->
            Sup.Redis.del("push_tokens:#{user_id}")
            {:reply, :ok, state}

          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
    end
  end

  # Private functions
  defp should_send_push?(user_id) do
    # Check if user is online
    case Sup.Presence.PresenceService.is_user_online?(user_id) do
      # User is online, no need for push
      true -> false
      # User is offline, send push
      false -> true
    end
  end

  defp get_user_push_token(user_id) do
    case Sup.Redis.hgetall("push_tokens:#{user_id}") do
      {:ok, []} ->
        {:error, "no_token"}

      {:ok, data} ->
        token =
          Enum.find_value(data, fn
            {"token", value} -> value
            _ -> nil
          end)

        platform =
          Enum.find_value(data, fn
            {"platform", value} -> value
            _ -> nil
          end)

        if token && platform do
          {:ok, token, platform}
        else
          {:error, "incomplete_token_data"}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp send_push_notification(token, platform, message, room_name, state) do
    case platform do
      "android" -> send_fcm_notification(token, message, room_name, state)
      "ios" -> send_apns_notification(token, message, room_name, state)
      "web" -> send_web_push_notification(token, message, room_name, state)
      _ -> Logger.warning("Unknown platform: #{platform}")
    end
  end

  defp send_fcm_notification(token, message, room_name, state) do
    if state.fcm_key do
      payload = %{
        to: token,
        notification: %{
          title: room_name,
          body: truncate_message(message.content),
          sound: "default"
        },
        data: %{
          room_id: message.room_id,
          message_id: message.id,
          sender_id: message.sender_id
        }
      }

      # Send HTTP request to FCM
      headers = [
        {"Authorization", "key=#{state.fcm_key}"},
        {"Content-Type", "application/json"}
      ]

      case HTTPoison.post("https://fcm.googleapis.com/fcm/send", Jason.encode!(payload), headers) do
        {:ok, %HTTPoison.Response{status_code: 200}} ->
          Logger.debug("FCM notification sent successfully")

        {:ok, %HTTPoison.Response{status_code: status_code, body: body}} ->
          Logger.warning("FCM notification failed: #{status_code} - #{body}")

        {:error, reason} ->
          Logger.error("FCM request failed: #{inspect(reason)}")
      end
    else
      Logger.warning("FCM server key not configured")
    end
  end

  defp send_apns_notification(token, _message, _room_name, _state) do
    # APNS implementation would go here
    # For now, just log
    Logger.debug("APNS notification would be sent to #{token}")
  end

  defp send_web_push_notification(token, _message, _room_name, _state) do
    # Web Push implementation would go here
    # For now, just log
    Logger.debug("Web Push notification would be sent to #{token}")
  end

  defp truncate_message(content, max_length \\ 100) do
    if String.length(content) > max_length do
      String.slice(content, 0, max_length) <> "..."
    else
      content
    end
  end
end
