defmodule Sup.Media.RichMediaService do
  @moduledoc """
  Service for handling rich media uploads, processing, and management.
  Supports images, videos, audio files, documents, and other attachments.
  """

  alias Sup.Media.MessageAttachment
  alias Sup.Repo
  import Ecto.Query
  require Logger

  @supported_image_types ["image/jpeg", "image/png", "image/gif", "image/webp"]
  @supported_video_types ["video/mp4", "video/webm", "video/mov", "video/avi"]
  @supported_audio_types ["audio/mp3", "audio/wav", "audio/ogg", "audio/m4a"]
  @supported_document_types ["application/pdf", "text/plain", "application/msword",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]

  @max_file_size_mb 50
  @max_file_size_bytes @max_file_size_mb * 1024 * 1024

  @doc """
  Upload and process a media file
  """
  def upload_media(user_id, file_data, opts \\ []) do
    with :ok <- validate_file(file_data),
         {:ok, processed_file} <- process_file(file_data),
         {:ok, attachment} <- create_attachment_record(user_id, processed_file, opts) do

      # Generate thumbnails for images/videos
      Task.start(fn -> generate_thumbnails(attachment) end)

      {:ok, MessageAttachment.public_fields(attachment)}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get attachment by ID
  """
  def get_attachment(attachment_id, user_id) do
    case Repo.get(MessageAttachment, attachment_id) do
      nil ->
        {:error, "attachment_not_found"}

      attachment ->
        if can_access_attachment?(user_id, attachment) do
          {:ok, MessageAttachment.public_fields(attachment)}
        else
          {:error, "unauthorized"}
        end
    end
  end

  @doc """
  Delete attachment
  """
  def delete_attachment(attachment_id, user_id) do
    case Repo.get(MessageAttachment, attachment_id) do
      nil ->
        {:error, "attachment_not_found"}

      attachment ->
        if can_delete_attachment?(user_id, attachment) do
          # Delete physical files
          delete_physical_files(attachment)

          # Delete database record
          case Repo.delete(attachment) do
            {:ok, _} ->
              {:ok, "attachment_deleted"}

            {:error, changeset} ->
              {:error, changeset}
          end
        else
          {:error, "unauthorized"}
        end
    end
  end

  @doc """
  Get attachments for a message
  """
  def get_message_attachments(message_id) do
    from(a in MessageAttachment, where: a.message_id == ^message_id)
    |> Repo.all()
    |> Enum.map(&MessageAttachment.public_fields/1)
  end

  @doc """
  Search attachments by type or metadata
  """
  def search_attachments(user_id, filters \\ %{}) do
    query = from(a in MessageAttachment, where: a.uploaded_by == ^user_id)

    query = apply_attachment_filters(query, filters)

    attachments = query
                 |> order_by([a], desc: a.inserted_at)
                 |> limit(50)
                 |> Repo.all()
                 |> Enum.map(&MessageAttachment.public_fields/1)

    {:ok, attachments}
  end

  @doc """
  Generate link preview for URLs
  """
  def generate_link_preview(url) do
    Task.start(fn ->
      case fetch_url_metadata(url) do
        {:ok, metadata} ->
          # Store link preview in cache or database
          cache_link_preview(url, metadata)

        {:error, reason} ->
          Logger.warning("Failed to generate link preview for #{url}: #{reason}")
      end
    end)

    {:ok, "preview_generation_started"}
  end

  @doc """
  Get cached link preview
  """
  def get_link_preview(url) do
    # This would check Redis cache or database
    case get_cached_link_preview(url) do
      nil -> {:error, "preview_not_found"}
      preview -> {:ok, preview}
    end
  end

  @doc """
  Process and optimize media files
  """
  def optimize_media(attachment_id) do
    case Repo.get(MessageAttachment, attachment_id) do
      nil ->
        {:error, "attachment_not_found"}

      attachment ->
        Task.start(fn -> perform_optimization(attachment) end)
        {:ok, "optimization_started"}
    end
  end

  # Private functions

  defp validate_file(%{"data" => data, "filename" => filename, "content_type" => content_type}) do
    with :ok <- validate_file_size(data),
         :ok <- validate_file_type(content_type),
         :ok <- validate_filename(filename) do
      :ok
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_file(_), do: {:error, "invalid_file_format"}

  defp validate_file_size(data) when byte_size(data) > @max_file_size_bytes do
    {:error, "file_too_large"}
  end
  defp validate_file_size(_), do: :ok

  defp validate_file_type(content_type) do
    allowed_types = @supported_image_types ++ @supported_video_types ++
                   @supported_audio_types ++ @supported_document_types

    if content_type in allowed_types do
      :ok
    else
      {:error, "unsupported_file_type"}
    end
  end

  defp validate_filename(filename) do
    if String.length(filename) > 255 do
      {:error, "filename_too_long"}
    else
      :ok
    end
  end

  defp process_file(%{"data" => data, "filename" => filename, "content_type" => content_type}) do
    # Generate unique filename
    unique_filename = generate_unique_filename(filename)
    file_path = get_upload_path(unique_filename)

    # Write file to storage
    case File.write(file_path, data) do
      :ok ->
        file_info = %{
          original_filename: filename,
          stored_filename: unique_filename,
          file_path: file_path,
          content_type: content_type,
          file_size: byte_size(data),
          checksum: generate_checksum(data)
        }

        {:ok, file_info}

      {:error, reason} ->
        {:error, "file_write_failed: #{reason}"}
    end
  end

  defp create_attachment_record(user_id, file_info, opts) do
    attachment_attrs = %{
      uploaded_by: user_id,
      message_id: Keyword.get(opts, :message_id),
      original_filename: file_info.original_filename,
      stored_filename: file_info.stored_filename,
      content_type: file_info.content_type,
      file_size: file_info.file_size,
      file_path: file_info.file_path,
      checksum: file_info.checksum,
      metadata: extract_file_metadata(file_info),
      processing_status: "completed"
    }

    case MessageAttachment.changeset(%MessageAttachment{}, attachment_attrs) |> Repo.insert() do
      {:ok, attachment} ->
        Logger.info("Created attachment #{attachment.id} for user #{user_id}")
        {:ok, attachment}

      {:error, changeset} ->
        # Clean up file if database insert fails
        File.rm(file_info.file_path)
        {:error, changeset}
    end
  end

  defp generate_thumbnails(attachment) do
    case attachment.content_type do
      "image/" <> _ ->
        generate_image_thumbnails(attachment)

      "video/" <> _ ->
        generate_video_thumbnails(attachment)

      _ ->
        :ok
    end
  end

  defp generate_image_thumbnails(attachment) do
    try do
      # Generate different sizes
      sizes = [
        {150, 150, "thumb"},
        {300, 300, "small"},
        {800, 600, "medium"}
      ]

      thumbnails = Enum.map(sizes, fn {width, height, size_name} ->
        generate_image_thumbnail(attachment, width, height, size_name)
      end)

      # Update attachment with thumbnail paths
      thumbnail_data = Enum.reduce(thumbnails, %{}, fn {size, path}, acc ->
        Map.put(acc, size, path)
      end)

      updated_metadata = Map.put(attachment.metadata || %{}, "thumbnails", thumbnail_data)

      attachment
      |> MessageAttachment.changeset(%{metadata: updated_metadata})
      |> Repo.update()

    rescue
      error ->
        Logger.error("Failed to generate thumbnails for #{attachment.id}: #{inspect(error)}")
    end
  end

  defp generate_image_thumbnail(attachment, width, height, size_name) do
    input_path = attachment.file_path
    output_filename = "#{Path.rootname(attachment.stored_filename)}_#{size_name}#{Path.extname(attachment.stored_filename)}"
    output_path = get_thumbnail_path(output_filename)

    # This would use ImageMagick or similar tool
    case System.cmd("convert", [
      input_path,
      "-resize", "#{width}x#{height}^",
      "-gravity", "center",
      "-extent", "#{width}x#{height}",
      output_path
    ]) do
      {_, 0} ->
        {size_name, output_path}

      {error, _} ->
        Logger.error("Thumbnail generation failed: #{error}")
        {size_name, nil}
    end
  end

  defp generate_video_thumbnails(attachment) do
    try do
      input_path = attachment.file_path
      output_filename = "#{Path.rootname(attachment.stored_filename)}_thumb.jpg"
      output_path = get_thumbnail_path(output_filename)

      # Extract thumbnail using ffmpeg
      case System.cmd("ffmpeg", [
        "-i", input_path,
        "-vf", "thumbnail,scale=300:200",
        "-frames:v", "1",
        output_path
      ]) do
        {_, 0} ->
          updated_metadata = Map.put(attachment.metadata || %{}, "video_thumbnail", output_path)

          attachment
          |> MessageAttachment.changeset(%{metadata: updated_metadata})
          |> Repo.update()

        {error, _} ->
          Logger.error("Video thumbnail generation failed: #{error}")
      end

    rescue
      error ->
        Logger.error("Failed to generate video thumbnail for #{attachment.id}: #{inspect(error)}")
    end
  end

  defp can_access_attachment?(user_id, attachment) do
    # Users can access their own attachments
    attachment.uploaded_by == user_id ||
    # Users can access attachments in rooms they're members of
    (attachment.message_id && can_access_message?(user_id, attachment.message_id))
  end

  defp can_delete_attachment?(user_id, attachment) do
    # Only the uploader can delete attachments
    attachment.uploaded_by == user_id
  end

  defp can_access_message?(user_id, message_id) do
    # This would check if user has access to the message's room
    # For now, return true as a placeholder
    true
  end

  defp delete_physical_files(attachment) do
    # Delete main file
    File.rm(attachment.file_path)

    # Delete thumbnails if they exist
    if attachment.metadata && attachment.metadata["thumbnails"] do
      Enum.each(attachment.metadata["thumbnails"], fn {_size, path} ->
        if path, do: File.rm(path)
      end)
    end

    # Delete video thumbnail if it exists
    if attachment.metadata && attachment.metadata["video_thumbnail"] do
      File.rm(attachment.metadata["video_thumbnail"])
    end
  end

  defp apply_attachment_filters(query, filters) do
    Enum.reduce(filters, query, fn
      {"content_type", content_type}, q ->
        where(q, [a], a.content_type == ^content_type)

      {"file_type", "image"}, q ->
        where(q, [a], like(a.content_type, "image/%"))

      {"file_type", "video"}, q ->
        where(q, [a], like(a.content_type, "video/%"))

      {"file_type", "audio"}, q ->
        where(q, [a], like(a.content_type, "audio/%"))

      {"file_type", "document"}, q ->
        where(q, [a], like(a.content_type, "application/%") or like(a.content_type, "text/%"))

      {"date_from", date}, q ->
        where(q, [a], a.inserted_at >= ^date)

      {"date_to", date}, q ->
        where(q, [a], a.inserted_at <= ^date)

      _, q -> q
    end)
  end

  defp extract_file_metadata(file_info) do
    base_metadata = %{
      "file_size_formatted" => format_file_size(file_info.file_size),
      "uploaded_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    case file_info.content_type do
      "image/" <> _ ->
        Map.merge(base_metadata, extract_image_metadata(file_info.file_path))

      "video/" <> _ ->
        Map.merge(base_metadata, extract_video_metadata(file_info.file_path))

      "audio/" <> _ ->
        Map.merge(base_metadata, extract_audio_metadata(file_info.file_path))

      _ ->
        base_metadata
    end
  end

  defp extract_image_metadata(file_path) do
    # This would use ExifRead or similar library
    %{
      "type" => "image",
      "dimensions" => get_image_dimensions(file_path)
    }
  end

  defp extract_video_metadata(file_path) do
    # This would use ffprobe or similar tool
    %{
      "type" => "video",
      "duration" => get_video_duration(file_path),
      "dimensions" => get_video_dimensions(file_path)
    }
  end

  defp extract_audio_metadata(file_path) do
    # This would use audio metadata library
    %{
      "type" => "audio",
      "duration" => get_audio_duration(file_path)
    }
  end

  # Placeholder implementations
  defp get_image_dimensions(_path), do: "1920x1080"
  defp get_video_duration(_path), do: "00:02:30"
  defp get_video_dimensions(_path), do: "1920x1080"
  defp get_audio_duration(_path), do: "00:03:45"

  defp fetch_url_metadata(url) do
    case HTTPoison.get(url, [], timeout: 5000, recv_timeout: 5000) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        parse_html_metadata(body, url)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_html_metadata(html, url) do
    # This would use Floki or similar HTML parser
    metadata = %{
      title: extract_title(html),
      description: extract_description(html),
      image: extract_image(html),
      url: url,
      site_name: extract_site_name(html),
      generated_at: DateTime.utc_now()
    }

    {:ok, metadata}
  end

  # Placeholder HTML parsing functions
  defp extract_title(_html), do: "Example Title"
  defp extract_description(_html), do: "Example description"
  defp extract_image(_html), do: "https://example.com/image.jpg"
  defp extract_site_name(_html), do: "Example Site"

  defp cache_link_preview(url, metadata) do
    # This would store in Redis with expiration
    # For now, just log it
    Logger.debug("Caching link preview for #{url}")
  end

  defp get_cached_link_preview(_url) do
    # This would retrieve from Redis cache
    # For now, return nil
    nil
  end

  defp perform_optimization(attachment) do
    Logger.info("Starting optimization for attachment #{attachment.id}")
    # Implementation would depend on file type
    # - Compress images
    # - Transcode videos
    # - Compress documents
  end

  defp generate_unique_filename(original_filename) do
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    random_suffix = :crypto.strong_rand_bytes(8) |> Base.encode64() |> binary_part(0, 11)
    extension = Path.extname(original_filename)

    "#{timestamp}_#{random_suffix}#{extension}"
  end

  defp get_upload_path(filename) do
    upload_dir = Application.get_env(:sup, :upload_directory, "uploads")
    File.mkdir_p!(upload_dir)
    Path.join(upload_dir, filename)
  end

  defp get_thumbnail_path(filename) do
    thumbnail_dir = Application.get_env(:sup, :thumbnail_directory, "uploads/thumbnails")
    File.mkdir_p!(thumbnail_dir)
    Path.join(thumbnail_dir, filename)
  end

  defp generate_checksum(data) do
    :crypto.hash(:sha256, data) |> Base.encode64()
  end

  defp format_file_size(size_bytes) do
    cond do
      size_bytes >= 1_073_741_824 -> "#{Float.round(size_bytes / 1_073_741_824, 2)} GB"
      size_bytes >= 1_048_576 -> "#{Float.round(size_bytes / 1_048_576, 2)} MB"
      size_bytes >= 1024 -> "#{Float.round(size_bytes / 1024, 2)} KB"
      true -> "#{size_bytes} bytes"
    end
  end
end
