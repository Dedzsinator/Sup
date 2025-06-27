defmodule Sup.Messaging.EnhancedMessageService do
  @moduledoc """
  Enhanced messaging service with Signal Protocol E2E encryption,
  threaded conversations, message reactions, and advanced features.
  """

  alias Sup.Messaging.{Message, DeliveryReceipt, SignalProtocol, MessageThread, MessageReaction}
  alias Sup.Messaging.{BotService, OfflineQueueService, CustomEmojiService}
  alias Sup.Analytics.AnalyticsService
  alias Sup.Media.RichMediaService
  alias Sup.Sync.MultiDeviceSyncService
  alias Sup.Room.RoomService
  alias Sup.SpamDetection.Service, as: SpamDetectionService
  alias Sup.Repo
  alias Sup.ScyllaDB
  alias Sup.Security.AuditLog
  import Ecto.Query
  require Logger

  @supported_message_types [:text, :image, :file, :audio, :video, :sticker, :location, :contact]

  # Enhanced message sending with E2E encryption
  def send_encrypted_message(
        sender_id,
        %{"room_id" => room_id, "content" => content, "type" => type} = params
      ) do
    case RoomService.can_send_message?(sender_id, room_id) do
      true ->
        recipients = get_room_recipients(room_id, sender_id)

        # Encrypt message for each recipient
        encrypted_messages =
          Enum.map(recipients, fn recipient_id ->
            case SignalProtocol.encrypt_message(sender_id, recipient_id, content) do
              {:ok, encrypted_message} ->
                encrypted_message

              {:error, "no_session"} ->
                # Establish session and retry
                establish_e2e_session(sender_id, recipient_id)

                {:ok, encrypted_message} =
                  SignalProtocol.encrypt_message(sender_id, recipient_id, content)

                encrypted_message
            end
          end)

        message_id = generate_message_id()
        timestamp = DateTime.utc_now()

        message_attrs = %{
          id: message_id,
          sender_id: sender_id,
          room_id: room_id,
          content: content,
          type: String.to_existing_atom(type),
          timestamp: timestamp,
          reply_to_id: Map.get(params, "reply_to_id"),
          thread_id: Map.get(params, "thread_id"),
          is_encrypted: true,
          encrypted_messages: encrypted_messages,
          metadata: build_message_metadata(params)
        }

        # Save to ScyllaDB for fast retrieval
        case ScyllaDB.insert_message(message_attrs) do
          :ok ->
            # Create delivery receipts for all room members
            create_delivery_receipts(message_id, room_id, sender_id)

            # Update thread if replying
            if params["reply_to_id"] do
              update_thread_activity(message_id, params["reply_to_id"])
            end

            # Process for bot commands
            BotService.process_message_for_bots(message_attrs)

            # Track analytics
            AnalyticsService.track_event(sender_id, "message_sent", %{
              room_id: room_id,
              encrypted: true,
              content_length: String.length(content),
              has_media: Map.has_key?(params, "media_metadata")
            })

            # Queue for offline users
            queue_for_offline_users(message_attrs)

            # Sync to other devices
            MultiDeviceSyncService.sync_message_data(sender_id, %{
              message: message_attrs,
              action: "sent"
            })

            # Process rich media if present
            if params["media_url"] do
              Task.start(fn ->
                RichMediaService.process_media_message(message_id, params["media_url"])
              end)
            end

            # Broadcast to room
            broadcast_message(message_attrs)

            # Audit log
            AuditLog.log_data_access("message", message_id, "create", sender_id, %{
              room_id: room_id,
              encrypted: true,
              content_length: String.length(content)
            })

            {:ok, message_attrs}

          {:error, reason} ->
            {:error, reason}
        end

      false ->
        {:error, "unauthorized"}
    end
  end

  # Send unencrypted message (for backward compatibility)
  def send_message(sender_id, params) do
    case RoomService.can_send_message?(sender_id, params["room_id"]) do
      true ->
        # Check for spam before processing the message
        case SpamDetectionService.process_message(
               params["content"],
               sender_id,
               params["room_id"]
             ) do
          {:error, :spam_detected, spam_info} ->
            Logger.warning("Spam message blocked from user #{sender_id}: #{inspect(spam_info)}")
            {:error, :spam_detected}

          {:ok, message_data} ->
            # Message was already processed and sent by spam detection service
            {:ok, message_data}

          # If spam detection returns the original content, continue with normal processing
          {:ok, _processed_content} ->
            process_normal_message(sender_id, params)

          # If spam detection fails, continue with normal processing (fail-open)
          _other ->
            Logger.warning("Spam detection failed for user #{sender_id}, proceeding with message")
            process_normal_message(sender_id, params)
        end

      false ->
        {:error, :unauthorized}
    end
  end

  defp process_normal_message(sender_id, params) do
    message_id = generate_message_id()
    timestamp = DateTime.utc_now()

    message_attrs = %{
      id: message_id,
      sender_id: sender_id,
      room_id: params["room_id"],
      content: params["content"],
      type: String.to_existing_atom(params["type"] || "text"),
      timestamp: timestamp,
      reply_to_id: Map.get(params, "reply_to_id"),
      thread_id: Map.get(params, "thread_id"),
      is_encrypted: false,
      metadata: build_message_metadata(params)
    }

    case ScyllaDB.insert_message(message_attrs) do
      :ok ->
        create_delivery_receipts(message_id, params["room_id"], sender_id)

        if params["reply_to_id"] do
          update_thread_activity(message_id, params["reply_to_id"])
        end

        # Process for bot commands
        BotService.process_message_for_bots(message_attrs)

        # Track analytics
        AnalyticsService.track_event(sender_id, "message_sent", %{
          room_id: params["room_id"],
          encrypted: false,
          content_length: String.length(params["content"]),
          message_type: params["type"]
        })

        # Queue for offline users
        queue_for_offline_users(message_attrs)

        # Sync to other devices
        MultiDeviceSyncService.sync_message_data(sender_id, %{
          message: message_attrs,
          action: "sent"
        })

        broadcast_message(message_attrs)
        {:ok, message_attrs}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Message reactions
  def add_reaction(user_id, message_id, emoji) do
    case can_react_to_message?(user_id, message_id) do
      true ->
        # Check if it's a custom emoji and validate
        if String.starts_with?(emoji, ":") and String.ends_with?(emoji, ":") do
          case CustomEmojiService.validate_emoji_usage(user_id, emoji) do
            {:error, reason} -> {:error, reason}
            :ok -> proceed_with_reaction(user_id, message_id, emoji)
          end
        else
          proceed_with_reaction(user_id, message_id, emoji)
        end

      false ->
        {:error, "unauthorized"}
    end
  end

  defp proceed_with_reaction(user_id, message_id, emoji) do
    reaction = %{
      id: Ecto.UUID.generate(),
      user_id: user_id,
      message_id: message_id,
      emoji: emoji,
      created_at: DateTime.utc_now()
    }

    case MessageReaction.changeset(%MessageReaction{}, reaction) |> Repo.insert() do
      {:ok, reaction} ->
        # Track analytics
        AnalyticsService.track_event(user_id, "reaction_added", %{
          message_id: message_id,
          emoji: emoji
        })

        # Track custom emoji usage
        if String.starts_with?(emoji, ":") do
          CustomEmojiService.track_emoji_usage(emoji, user_id)
        end

        # Sync to other devices
        MultiDeviceSyncService.sync_message_data(user_id, %{
          reaction: reaction,
          action: "added"
        })

        # Queue for offline users
        message = get_message(message_id)

        OfflineQueueService.queue_message(message.sender_id, %{
          type: "reaction",
          message_id: message_id,
          emoji: emoji,
          user_id: user_id,
          action: "added"
        })

        broadcast_reaction(reaction, "added")
        {:ok, reaction}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  def remove_reaction(user_id, message_id, emoji) do
    case Repo.get_by(MessageReaction, user_id: user_id, message_id: message_id, emoji: emoji) do
      nil ->
        {:error, "reaction_not_found"}

      reaction ->
        case Repo.delete(reaction) do
          {:ok, reaction} ->
            # Track analytics
            AnalyticsService.track_event(user_id, "reaction_removed", %{
              message_id: message_id,
              emoji: emoji
            })

            # Sync to other devices
            MultiDeviceSyncService.sync_message_data(user_id, %{
              reaction: reaction,
              action: "removed"
            })

            broadcast_reaction(reaction, "removed")
            {:ok, reaction}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  # Message threading
  def create_thread(user_id, parent_message_id, %{"content" => content, "type" => type} = params) do
    case can_reply_to_message?(user_id, parent_message_id) do
      true ->
        parent_message = get_message(parent_message_id)
        thread_id = parent_message.thread_id || parent_message_id

        thread_params =
          Map.merge(params, %{
            "room_id" => parent_message.room_id,
            "thread_id" => thread_id,
            "reply_to_id" => parent_message_id
          })

        case send_message(user_id, thread_params) do
          {:ok, message} ->
            # Create or update thread metadata
            upsert_thread_metadata(thread_id, parent_message.room_id, user_id)

            # Track analytics
            AnalyticsService.track_event(user_id, "thread_message_sent", %{
              thread_id: thread_id,
              parent_message_id: parent_message_id,
              room_id: parent_message.room_id
            })

            {:ok, message}

          error ->
            error
        end

      false ->
        {:error, "unauthorized"}
    end
  end

  def get_thread_messages(user_id, thread_id, limit \\ 50, before_timestamp \\ nil) do
    case can_access_thread?(user_id, thread_id) do
      true ->
        ScyllaDB.get_thread_messages(thread_id, limit, before_timestamp)

      false ->
        {:error, "unauthorized"}
    end
  end

  # Message search with mentions
  def search_messages(user_id, query, %{"room_id" => room_id} = filters) do
    case RoomService.can_access_room?(user_id, room_id) do
      true ->
        search_filters = %{
          room_ids: [room_id],
          user_id: Map.get(filters, "user_id"),
          message_type: Map.get(filters, "type"),
          date_from: Map.get(filters, "date_from"),
          date_to: Map.get(filters, "date_to"),
          has_attachments: Map.get(filters, "has_attachments"),
          mentions_user: Map.get(filters, "mentions_user")
        }

        ScyllaDB.search_messages(query, search_filters)

      false ->
        {:error, "unauthorized"}
    end
  end

  def search_mentions(user_id, limit \\ 20) do
    user_rooms = RoomService.get_user_rooms(user_id)
    room_ids = Enum.map(user_rooms, & &1.id)

    ScyllaDB.search_mentions(user_id, room_ids, limit)
  end

  # Message editing and deletion
  def edit_message(user_id, message_id, new_content) do
    case can_edit_message?(user_id, message_id) do
      true ->
        timestamp = DateTime.utc_now()

        case ScyllaDB.update_message(message_id, %{
               content: new_content,
               edited_at: timestamp
             }) do
          :ok ->
            # Broadcast edit
            broadcast_message_edit(message_id, new_content, timestamp)

            # Audit log
            AuditLog.log_data_modification(
              "message",
              message_id,
              "edit",
              %{
                new_content_length: String.length(new_content)
              },
              user_id,
              %{}
            )

            {:ok, %{message_id: message_id, edited_at: timestamp}}

          {:error, reason} ->
            {:error, reason}
        end

      false ->
        {:error, "unauthorized"}
    end
  end

  def delete_message(user_id, message_id) do
    case can_delete_message?(user_id, message_id) do
      true ->
        case ScyllaDB.delete_message(message_id) do
          :ok ->
            # Remove reactions
            from(r in MessageReaction, where: r.message_id == ^message_id)
            |> Repo.delete_all()

            # Broadcast deletion
            broadcast_message_deletion(message_id)

            # Audit log
            AuditLog.log_data_modification("message", message_id, "delete", %{}, user_id, %{})

            {:ok, message_id}

          {:error, reason} ->
            {:error, reason}
        end

      false ->
        {:error, "unauthorized"}
    end
  end

  # Enhanced delivery receipts
  def mark_message_read(message_id, user_id) do
    case Repo.get_by(DeliveryReceipt, message_id: message_id, user_id: user_id) do
      nil ->
        {:error, "receipt_not_found"}

      receipt ->
        changeset =
          DeliveryReceipt.changeset(receipt, %{
            read_at: DateTime.utc_now(),
            status: :read
          })

        case Repo.update(changeset) do
          {:ok, updated_receipt} ->
            # Broadcast read receipt to sender
            broadcast_delivery_receipt(updated_receipt)

            # Update thread read status if applicable
            if receipt.thread_id do
              update_thread_read_status(receipt.thread_id, user_id)
            end

            # Track analytics
            AnalyticsService.track_event(user_id, "message_read", %{
              message_id: message_id
            })

            # Sync to other devices
            MultiDeviceSyncService.sync_message_data(user_id, %{
              read_receipt: updated_receipt,
              action: "read"
            })

            {:ok, updated_receipt}

          {:error, changeset} ->
            {:error, changeset}
        end
    end
  end

  def mark_thread_read(thread_id, user_id) do
    # Mark all messages in thread as read
    thread_messages = ScyllaDB.get_thread_messages(thread_id, 1000)

    Enum.each(thread_messages, fn message ->
      mark_message_read(message.id, user_id)
    end)

    {:ok, thread_id}
  end

  # Private functions
  defp establish_e2e_session(sender_id, recipient_id) do
    key_bundle = SignalProtocol.get_key_bundle(recipient_id)
    SignalProtocol.establish_session(sender_id, recipient_id, key_bundle)
  end

  defp get_room_recipients(room_id, sender_id) do
    RoomService.get_room_members(room_id)
    |> Enum.reject(&(&1.id == sender_id))
    |> Enum.map(& &1.id)
  end

  defp build_message_metadata(params) do
    %{
      mentions: extract_mentions(params["content"] || ""),
      hashtags: extract_hashtags(params["content"] || ""),
      links: extract_links(params["content"] || ""),
      media_metadata: Map.get(params, "media_metadata", %{}),
      location: Map.get(params, "location"),
      quoted_message_id: Map.get(params, "quoted_message_id")
    }
  end

  defp extract_mentions(content) do
    Regex.scan(~r/@(\w+)/, content, capture: :all_but_first)
    |> List.flatten()
  end

  defp extract_hashtags(content) do
    Regex.scan(~r/#(\w+)/, content, capture: :all_but_first)
    |> List.flatten()
  end

  defp extract_links(content) do
    Regex.scan(~r/https?:\/\/[^\s]+/, content)
    |> List.flatten()
  end

  defp update_thread_activity(message_id, parent_message_id) do
    # Update thread with latest activity
    thread_id = get_thread_id(parent_message_id)

    MessageThread.changeset(%MessageThread{}, %{
      thread_id: thread_id,
      last_message_id: message_id,
      last_activity_at: DateTime.utc_now()
    })
    |> Repo.insert_or_update()
  end

  defp upsert_thread_metadata(thread_id, room_id, user_id) do
    thread_attrs = %{
      id: thread_id,
      room_id: room_id,
      created_by: user_id,
      message_count: get_thread_message_count(thread_id),
      participants: get_thread_participants(thread_id),
      last_activity_at: DateTime.utc_now()
    }

    MessageThread.changeset(%MessageThread{}, thread_attrs)
    |> Repo.insert_or_update()
  end

  defp can_react_to_message?(user_id, message_id) do
    message = get_message(message_id)
    message && RoomService.can_access_room?(user_id, message.room_id)
  end

  defp can_reply_to_message?(user_id, message_id) do
    message = get_message(message_id)
    message && RoomService.can_send_message?(user_id, message.room_id)
  end

  defp can_edit_message?(user_id, message_id) do
    message = get_message(message_id)

    message &&
      (message.sender_id == user_id || RoomService.is_room_admin?(user_id, message.room_id))
  end

  defp can_delete_message?(user_id, message_id) do
    message = get_message(message_id)

    message &&
      (message.sender_id == user_id || RoomService.is_room_admin?(user_id, message.room_id))
  end

  defp can_access_thread?(user_id, thread_id) do
    thread = Repo.get(MessageThread, thread_id)
    thread && RoomService.can_access_room?(user_id, thread.room_id)
  end

  defp get_message(message_id) do
    ScyllaDB.get_message(message_id)
  end

  defp get_thread_id(message_id) do
    message = get_message(message_id)
    message.thread_id || message_id
  end

  defp get_thread_message_count(thread_id) do
    ScyllaDB.count_thread_messages(thread_id)
  end

  defp get_thread_participants(thread_id) do
    ScyllaDB.get_thread_participants(thread_id)
  end

  defp update_thread_read_status(thread_id, user_id) do
    # Implementation for tracking thread read status
    :ok
  end

  # Broadcasting functions
  defp broadcast_message(message) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{message.room_id}", {:message, message})
  end

  defp broadcast_reaction(reaction, action) do
    message = get_message(reaction.message_id)

    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{message.room_id}", {
      :reaction,
      %{
        action: action,
        message_id: reaction.message_id,
        user_id: reaction.user_id,
        emoji: reaction.emoji
      }
    })
  end

  defp broadcast_message_edit(message_id, new_content, edited_at) do
    message = get_message(message_id)

    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{message.room_id}", {
      :message_edited,
      %{
        message_id: message_id,
        content: new_content,
        edited_at: edited_at
      }
    })
  end

  defp broadcast_message_deletion(message_id) do
    message = get_message(message_id)

    Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{message.room_id}", {
      :message_deleted,
      %{message_id: message_id}
    })
  end

  defp broadcast_delivery_receipt(receipt) do
    Phoenix.PubSub.broadcast(Sup.PubSub, "user:#{receipt.user_id}", {:delivery_receipt, receipt})
  end

  defp create_delivery_receipts(message_id, room_id, sender_id) do
    # Get all room members except sender
    members =
      RoomService.get_room_members(room_id)
      |> Enum.reject(&(&1.id == sender_id))

    receipts =
      Enum.map(members, fn member ->
        %{
          id: Ecto.UUID.generate(),
          message_id: message_id,
          user_id: member.id,
          status: :sent,
          sent_at: DateTime.utc_now()
        }
      end)

    Repo.insert_all(DeliveryReceipt, receipts)
  end

  defp queue_for_offline_users(message) do
    room_members = RoomService.get_room_members(message.room_id)

    Enum.each(room_members, fn member ->
      unless member.id == message.sender_id do
        OfflineQueueService.queue_message(member.id, %{
          type: "message",
          id: message.id,
          room_id: message.room_id,
          sender_id: message.sender_id,
          content: message.content,
          timestamp: message.timestamp,
          metadata: message.metadata || %{}
        })
      end
    end)
  end

  defp generate_message_id do
    Ecto.UUID.generate()
  end
end
