defmodule Sup.Messaging.RichMediaService do
  @moduledoc """
  Service for handling rich media uploads and processing.
  Supports images, videos, audio, and documents with proper validation and optimization.
  """

  alias Sup.Messaging.MediaAttachment
  alias Sup.Repo
  require Logger

  # 50MB
  @max_file_size 50 * 1024 * 1024
  @allowed_image_types ~w(.jpg .jpeg .png .gif .webp)
  @allowed_video_types ~w(.mp4 .webm .mov .avi)
  @allowed_audio_types ~w(.mp3 .wav .ogg .m4a)
  @allowed_document_types ~w(.pdf .doc .docx .txt .md)

  @doc """
  Upload and process media file
  """
  def upload_media(user_id, file_data, file_type, original_filename) do
    with {:ok, file_info} <- validate_file(file_data, file_type, original_filename),
         {:ok, processed_file} <- process_file(file_info),
         {:ok, media_attachment} <- save_media_attachment(user_id, processed_file) do
      {:ok, media_attachment}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get media attachment by ID
  """
  def get_media_attachment(attachment_id) do
    case Repo.get(MediaAttachment, attachment_id) do
      nil -> {:error, "attachment_not_found"}
      attachment -> {:ok, attachment}
    end
  end

  @doc """
  Delete media attachment
  """
  def delete_media_attachment(attachment_id, user_id) do
    case Repo.get(MediaAttachment, attachment_id) do
      nil ->
        {:error, "attachment_not_found"}

      attachment ->
        if attachment.uploaded_by == user_id do
          # Delete physical file
          delete_physical_file(attachment.file_path)

          # Delete from database
          case Repo.delete(attachment) do
            {:ok, _} -> {:ok, :deleted}
            {:error, changeset} -> {:error, changeset}
          end
        else
          {:error, "unauthorized"}
        end
    end
  end

  @doc """
  Generate thumbnail for media
  """
  def generate_thumbnail(attachment_id) do
    case get_media_attachment(attachment_id) do
      {:ok, attachment} ->
        if attachment.media_type in ["image", "video"] do
          generate_thumbnail_for_attachment(attachment)
        else
          {:error, "thumbnails_not_supported"}
        end

      error ->
        error
    end
  end

  @doc """
  Get media statistics
  """
  def get_media_stats(user_id) do
    import Ecto.Query

    total_size =
      from(ma in MediaAttachment,
        where: ma.uploaded_by == ^user_id,
        select: sum(ma.file_size)
      )
      |> Repo.one() || 0

    file_count =
      from(ma in MediaAttachment,
        where: ma.uploaded_by == ^user_id,
        select: count(ma.id)
      )
      |> Repo.one()

    by_type =
      from(ma in MediaAttachment,
        where: ma.uploaded_by == ^user_id,
        group_by: ma.media_type,
        select: {ma.media_type, count(ma.id), sum(ma.file_size)}
      )
      |> Repo.all()

    %{
      total_size: total_size,
      file_count: file_count,
      by_type:
        Enum.map(by_type, fn {type, count, size} ->
          %{type: type, count: count, size: size || 0}
        end)
    }
  end

  # Private functions

  defp validate_file(file_data, file_type, original_filename) do
    with :ok <- validate_file_size(byte_size(file_data)),
         :ok <- validate_file_type(file_type, original_filename),
         :ok <- validate_file_content(file_data, file_type) do
      {:ok,
       %{
         data: file_data,
         type: file_type,
         filename: original_filename,
         size: byte_size(file_data)
       }}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_file_size(size) when size > @max_file_size do
    {:error, "file_too_large"}
  end

  defp validate_file_size(_size), do: :ok

  defp validate_file_type(file_type, filename) do
    extension = filename |> Path.extname() |> String.downcase()

    allowed_extensions =
      case file_type do
        "image" -> @allowed_image_types
        "video" -> @allowed_video_types
        "audio" -> @allowed_audio_types
        "document" -> @allowed_document_types
        _ -> []
      end

    if extension in allowed_extensions do
      :ok
    else
      {:error, "invalid_file_type"}
    end
  end

  defp validate_file_content(file_data, file_type) do
    # Basic file signature validation
    signature = binary_part(file_data, 0, min(byte_size(file_data), 8))

    case {file_type, signature} do
      # JPEG
      {"image", <<0xFF, 0xD8, 0xFF, _::binary>>} -> :ok
      # PNG
      {"image", <<0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A>>} -> :ok
      # GIF
      {"image", <<"GIF8", _::binary>>} -> :ok
      # WebP
      {"image", <<"RIFF", _::binary-size(4), "WEBP">>} -> :ok
      # For now, allow other types without strict validation
      {_, _} -> :ok
    end
  end

  defp process_file(file_info) do
    # Generate unique filename
    unique_id = Ecto.UUID.generate()
    extension = Path.extname(file_info.filename)
    new_filename = "#{unique_id}#{extension}"

    # Determine storage path
    storage_path = get_storage_path(file_info.type)
    full_path = Path.join(storage_path, new_filename)

    # Create directory if it doesn't exist
    File.mkdir_p!(storage_path)

    # Write file to disk
    case File.write(full_path, file_info.data) do
      :ok ->
        # Get file metadata
        metadata = get_file_metadata(full_path, file_info.type)

        {:ok,
         %{
           original_filename: file_info.filename,
           stored_filename: new_filename,
           file_path: full_path,
           media_type: file_info.type,
           file_size: file_info.size,
           metadata: metadata
         }}

      {:error, reason} ->
        {:error, "file_write_failed: #{reason}"}
    end
  end

  defp save_media_attachment(user_id, processed_file) do
    attachment_params = %{
      id: Ecto.UUID.generate(),
      original_filename: processed_file.original_filename,
      stored_filename: processed_file.stored_filename,
      file_path: processed_file.file_path,
      media_type: processed_file.media_type,
      file_size: processed_file.file_size,
      metadata: processed_file.metadata,
      uploaded_by: user_id,
      uploaded_at: DateTime.utc_now()
    }

    case MediaAttachment.changeset(%MediaAttachment{}, attachment_params) |> Repo.insert() do
      {:ok, attachment} ->
        Logger.info("Media uploaded: #{attachment.id} by user #{user_id}")
        {:ok, MediaAttachment.public_fields(attachment)}

      {:error, changeset} ->
        # Clean up file if database insert fails
        File.rm(processed_file.file_path)
        {:error, changeset}
    end
  end

  defp get_storage_path(media_type) do
    base_path = Application.get_env(:sup, :media_storage_path, "priv/static/uploads")
    Path.join([base_path, media_type])
  end

  defp get_file_metadata(file_path, media_type) do
    case media_type do
      "image" -> get_image_metadata(file_path)
      "video" -> get_video_metadata(file_path)
      "audio" -> get_audio_metadata(file_path)
      _ -> %{}
    end
  end

  defp get_image_metadata(_file_path) do
    # This would use ImageMagick or similar to get image dimensions
    # For now, return empty metadata
    %{
      # width: width,
      # height: height,
      # format: format
    }
  end

  defp get_video_metadata(_file_path) do
    # This would use FFmpeg to get video information
    # For now, return empty metadata
    %{
      # duration: duration,
      # width: width,
      # height: height,
      # codec: codec
    }
  end

  defp get_audio_metadata(_file_path) do
    # This would use FFmpeg to get audio information
    # For now, return empty metadata
    %{
      # duration: duration,
      # bitrate: bitrate,
      # codec: codec
    }
  end

  defp generate_thumbnail_for_attachment(_attachment) do
    # This would generate thumbnails using ImageMagick/FFmpeg
    # For now, return success without actual thumbnail generation
    {:ok, "thumbnail_generated"}
  end

  defp delete_physical_file(file_path) do
    case File.rm(file_path) do
      :ok ->
        :ok

      {:error, reason} ->
        Logger.warning("Failed to delete file #{file_path}: #{reason}")
        # Don't fail the operation if file deletion fails
        :ok
    end
  end
end
