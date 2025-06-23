defmodule Sup.Messaging.CustomEmojiService do
  @moduledoc """
  Service for managing custom emojis, including upload, processing,
  and room-specific emoji collections.
  """

  alias Sup.Messaging.CustomEmoji
  alias Sup.Room.RoomService
  alias Sup.Repo
  import Ecto.Query
  require Logger

  @supported_image_types ["image/png", "image/gif", "image/webp"]
  @max_emoji_size_kb 256
  @max_emoji_size_bytes @max_emoji_size_kb * 1024
  @max_emoji_dimension 128

  @doc """
  Upload and create a custom emoji
  """
  def create_custom_emoji(user_id, emoji_data) do
    with :ok <- validate_emoji_data(emoji_data),
         :ok <- validate_user_permissions(user_id, emoji_data["room_id"]),
         {:ok, processed_emoji} <- process_emoji_file(emoji_data),
         {:ok, emoji} <- create_emoji_record(user_id, processed_emoji, emoji_data) do

      # Broadcast emoji creation to room
      broadcast_emoji_event(emoji_data["room_id"], :emoji_created, emoji)

      {:ok, CustomEmoji.public_fields(emoji)}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get custom emojis for a room
  """
  def get_room_emojis(room_id, user_id) do
    case RoomService.can_access_room?(user_id, room_id) do
      true ->
        emojis = from(e in CustomEmoji,
                     where: e.room_id == ^room_id and e.is_active == true,
                     order_by: [asc: e.name])
                |> Repo.all()
                |> Enum.map(&CustomEmoji.public_fields/1)

        {:ok, emojis}

      false ->
        {:error, "unauthorized"}
    end
  end

  @doc """
  Get global custom emojis (available to all users)
  """
  def get_global_emojis do
    emojis = from(e in CustomEmoji,
                 where: is_nil(e.room_id) and e.is_active == true,
                 order_by: [asc: e.name])
            |> Repo.all()
            |> Enum.map(&CustomEmoji.public_fields/1)

    {:ok, emojis}
  end

  @doc """
  Search emojis by name or tags
  """
  def search_emojis(query, user_id, room_id \\ nil) do
    search_term = "%#{String.downcase(query)}%"

    base_query = from(e in CustomEmoji,
                     where: e.is_active == true and
                            (ilike(e.name, ^search_term) or
                             fragment("? @> ?", e.tags, ^[String.downcase(query)])),
                     order_by: [asc: e.name],
                     limit: 20)

    query = if room_id do
      # Include room-specific and global emojis
      from(e in base_query,
           where: e.room_id == ^room_id or is_nil(e.room_id))
    else
      # Only global emojis
      from(e in base_query, where: is_nil(e.room_id))
    end

    emojis = query
            |> Repo.all()
            |> Enum.map(&CustomEmoji.public_fields/1)

    {:ok, emojis}
  end

  @doc """
  Update custom emoji
  """
  def update_emoji(emoji_id, user_id, updates) do
    case get_emoji_with_permissions(emoji_id, user_id) do
      {:ok, emoji} ->
        case CustomEmoji.changeset(emoji, updates) |> Repo.update() do
          {:ok, updated_emoji} ->
            broadcast_emoji_event(emoji.room_id, :emoji_updated, updated_emoji)
            {:ok, CustomEmoji.public_fields(updated_emoji)}

          {:error, changeset} ->
            {:error, changeset}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Delete custom emoji
  """
  def delete_emoji(emoji_id, user_id) do
    case get_emoji_with_permissions(emoji_id, user_id) do
      {:ok, emoji} ->
        # Soft delete
        case CustomEmoji.changeset(emoji, %{is_active: false}) |> Repo.update() do
          {:ok, _} ->
            # Delete physical file
            delete_emoji_file(emoji.file_path)

            broadcast_emoji_event(emoji.room_id, :emoji_deleted, %{id: emoji_id})
            {:ok, "emoji_deleted"}

          {:error, changeset} ->
            {:error, changeset}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Get emoji usage statistics
  """
  def get_emoji_stats(emoji_id, user_id) do
    case get_emoji_with_permissions(emoji_id, user_id) do
      {:ok, emoji} ->
        stats = %{
          usage_count: get_emoji_usage_count(emoji_id),
          recent_usage: get_recent_emoji_usage(emoji_id),
          top_users: get_emoji_top_users(emoji_id),
          usage_by_day: get_emoji_usage_by_day(emoji_id)
        }

        {:ok, stats}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Get popular emojis for a room
  """
  def get_popular_emojis(room_id, user_id, opts \\ []) do
    case RoomService.can_access_room?(user_id, room_id) do
      true ->
        limit = Keyword.get(opts, :limit, 10)
        days = Keyword.get(opts, :days, 7)

        popular_emojis = get_popular_emojis_impl(room_id, limit, days)
        {:ok, popular_emojis}

      false ->
        {:error, "unauthorized"}
    end
  end

  @doc """
  Bulk import emojis from an emoji pack
  """
  def import_emoji_pack(user_id, room_id, pack_data) do
    case RoomService.is_room_admin?(user_id, room_id) do
      true ->
        results = Enum.map(pack_data["emojis"], fn emoji_data ->
          import_single_emoji(user_id, room_id, emoji_data)
        end)

        success_count = Enum.count(results, fn {status, _} -> status == :ok end)
        error_count = length(results) - success_count

        {:ok, %{
          imported: success_count,
          errors: error_count,
          results: results
        }}

      false ->
        {:error, "unauthorized"}
    end
  end

  @doc """
  Export room emojis as a pack
  """
  def export_emoji_pack(room_id, user_id) do
    case RoomService.is_room_admin?(user_id, room_id) do
      true ->
        emojis = from(e in CustomEmoji,
                     where: e.room_id == ^room_id and e.is_active == true)
                |> Repo.all()

        pack_data = %{
          name: "Room Emoji Pack",
          description: "Custom emojis from room",
          emojis: Enum.map(emojis, &export_emoji_data/1),
          exported_at: DateTime.utc_now(),
          version: "1.0"
        }

        {:ok, pack_data}

      false ->
        {:error, "unauthorized"}
    end
  end

  # Private functions

  defp validate_emoji_data(emoji_data) do
    required_fields = ["name", "file_data", "content_type"]

    missing_fields = Enum.filter(required_fields, fn field ->
      not Map.has_key?(emoji_data, field) or is_nil(emoji_data[field])
    end)

    if length(missing_fields) > 0 do
      {:error, "missing_fields: #{Enum.join(missing_fields, ", ")}"}
    else
      with :ok <- validate_emoji_name(emoji_data["name"]),
           :ok <- validate_emoji_file(emoji_data["file_data"], emoji_data["content_type"]) do
        :ok
      else
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp validate_emoji_name(name) do
    cond do
      String.length(name) < 2 ->
        {:error, "emoji_name_too_short"}

      String.length(name) > 32 ->
        {:error, "emoji_name_too_long"}

      not Regex.match?(~r/^[a-zA-Z0-9_-]+$/, name) ->
        {:error, "emoji_name_invalid_characters"}

      true ->
        :ok
    end
  end

  defp validate_emoji_file(file_data, content_type) do
    cond do
      content_type not in @supported_image_types ->
        {:error, "unsupported_file_type"}

      byte_size(file_data) > @max_emoji_size_bytes ->
        {:error, "file_too_large"}

      true ->
        validate_emoji_dimensions(file_data)
    end
  end

  defp validate_emoji_dimensions(file_data) do
    # This would use an image processing library to check dimensions
    # For now, assume it's valid
    :ok
  end

  defp validate_user_permissions(user_id, room_id) do
    cond do
      is_nil(room_id) ->
        # Global emoji - check if user is admin
        if is_admin_user?(user_id) do
          :ok
        else
          {:error, "admin_required_for_global_emoji"}
        end

      true ->
        # Room emoji - check if user is room admin
        if RoomService.is_room_admin?(user_id, room_id) do
          :ok
        else
          {:error, "room_admin_required"}
        end
    end
  end

  defp process_emoji_file(emoji_data) do
    # Generate unique filename
    file_extension = get_file_extension(emoji_data["content_type"])
    unique_filename = "emoji_#{generate_unique_id()}#{file_extension}"
    file_path = get_emoji_storage_path(unique_filename)

    # Write file to storage
    case File.write(file_path, emoji_data["file_data"]) do
      :ok ->
        # Optimize and resize if needed
        optimized_path = optimize_emoji_file(file_path, emoji_data["content_type"])

        {:ok, %{
          original_filename: emoji_data["filename"] || unique_filename,
          stored_filename: unique_filename,
          file_path: optimized_path || file_path,
          content_type: emoji_data["content_type"],
          file_size: byte_size(emoji_data["file_data"]),
          checksum: generate_file_checksum(emoji_data["file_data"])
        }}

      {:error, reason} ->
        {:error, "file_write_failed: #{reason}"}
    end
  end

  defp create_emoji_record(user_id, processed_file, emoji_data) do
    emoji_attrs = %{
      name: String.downcase(emoji_data["name"]),
      display_name: emoji_data["name"],
      room_id: emoji_data["room_id"],
      created_by: user_id,
      file_path: processed_file.file_path,
      file_size: processed_file.file_size,
      content_type: processed_file.content_type,
      checksum: processed_file.checksum,
      tags: extract_emoji_tags(emoji_data),
      description: emoji_data["description"],
      is_active: true,
      usage_count: 0
    }

    case CustomEmoji.changeset(%CustomEmoji{}, emoji_attrs) |> Repo.insert() do
      {:ok, emoji} ->
        Logger.info("Created custom emoji #{emoji.name} for user #{user_id}")
        {:ok, emoji}

      {:error, changeset} ->
        # Clean up file if database insert fails
        File.rm(processed_file.file_path)
        {:error, changeset}
    end
  end

  defp get_emoji_with_permissions(emoji_id, user_id) do
    case Repo.get(CustomEmoji, emoji_id) do
      nil ->
        {:error, "emoji_not_found"}

      emoji ->
        if can_modify_emoji?(user_id, emoji) do
          {:ok, emoji}
        else
          {:error, "unauthorized"}
        end
    end
  end

  defp can_modify_emoji?(user_id, emoji) do
    cond do
      # Creator can always modify
      emoji.created_by == user_id -> true

      # Global emoji requires admin
      is_nil(emoji.room_id) -> is_admin_user?(user_id)

      # Room emoji requires room admin
      true -> RoomService.is_room_admin?(user_id, emoji.room_id)
    end
  end

  defp get_popular_emojis_impl(room_id, limit, days) do
    # This would query message analytics or emoji usage tracking
    # For now, return empty list
    []
  end

  defp get_emoji_usage_count(emoji_id) do
    # This would query message reactions or emoji usage analytics
    # For now, return 0
    0
  end

  defp get_recent_emoji_usage(emoji_id) do
    # This would get recent usage events
    []
  end

  defp get_emoji_top_users(emoji_id) do
    # This would get users who use the emoji most
    []
  end

  defp get_emoji_usage_by_day(emoji_id) do
    # This would get daily usage statistics
    %{}
  end

  defp import_single_emoji(user_id, room_id, emoji_data) do
    try do
      full_emoji_data = Map.put(emoji_data, "room_id", room_id)
      create_custom_emoji(user_id, full_emoji_data)
    rescue
      error ->
        {:error, inspect(error)}
    end
  end

  defp export_emoji_data(emoji) do
    %{
      name: emoji.name,
      display_name: emoji.display_name,
      description: emoji.description,
      tags: emoji.tags,
      file_data: read_emoji_file(emoji.file_path),
      content_type: emoji.content_type
    }
  end

  defp read_emoji_file(file_path) do
    case File.read(file_path) do
      {:ok, data} -> Base.encode64(data)
      {:error, _} -> nil
    end
  end

  defp optimize_emoji_file(file_path, content_type) do
    case content_type do
      "image/png" -> optimize_png(file_path)
      "image/gif" -> optimize_gif(file_path)
      "image/webp" -> optimize_webp(file_path)
      _ -> nil
    end
  end

  defp optimize_png(file_path) do
    # Use pngquant or similar tool to optimize PNG
    optimized_path = String.replace(file_path, ".png", "_optimized.png")

    case System.cmd("pngquant", ["--force", "--output", optimized_path, file_path]) do
      {_, 0} -> optimized_path
      _ -> nil
    end
  rescue
    _ -> nil
  end

  defp optimize_gif(file_path) do
    # GIFs are harder to optimize, skip for now
    nil
  end

  defp optimize_webp(file_path) do
    # WebP is already optimized
    nil
  end

  defp extract_emoji_tags(emoji_data) do
    tags = emoji_data["tags"] || []

    # Add automatic tags based on name
    automatic_tags = emoji_data["name"]
                    |> String.split(~r/[_-]/)
                    |> Enum.map(&String.downcase/1)
                    |> Enum.reject(&(String.length(&1) < 2))

    (tags ++ automatic_tags)
    |> Enum.uniq()
    |> Enum.take(10)  # Limit to 10 tags
  end

  defp delete_emoji_file(file_path) do
    case File.rm(file_path) do
      :ok -> :ok
      {:error, reason} ->
        Logger.error("Failed to delete emoji file #{file_path}: #{reason}")
    end
  end

  defp broadcast_emoji_event(room_id, event_type, emoji_data) do
    if room_id do
      Phoenix.PubSub.broadcast(Sup.PubSub, "room:#{room_id}", {
        event_type,
        emoji_data
      })
    else
      # Global emoji - broadcast to all users
      Phoenix.PubSub.broadcast(Sup.PubSub, "global_emojis", {
        event_type,
        emoji_data
      })
    end
  end

  defp is_admin_user?(user_id) do
    # This would check if user has admin role
    # For now, return false
    false
  end

  defp get_file_extension("image/png"), do: ".png"
  defp get_file_extension("image/gif"), do: ".gif"
  defp get_file_extension("image/webp"), do: ".webp"
  defp get_file_extension(_), do: ".png"

  defp get_emoji_storage_path(filename) do
    storage_dir = Application.get_env(:sup, :emoji_storage_directory, "uploads/emojis")
    File.mkdir_p!(storage_dir)
    Path.join(storage_dir, filename)
  end

  defp generate_unique_id do
    :crypto.strong_rand_bytes(8) |> Base.encode64() |> binary_part(0, 11)
  end

  defp generate_file_checksum(data) do
    :crypto.hash(:sha256, data) |> Base.encode64()
  end
end
