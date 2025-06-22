defmodule Sup.Security.TwoFactor do
  @moduledoc """
  Two-Factor Authentication functionality using TOTP (Time-based One-Time Password).
  """

  alias Sup.Auth.User
  alias Sup.Repo
  alias Sup.Security.Config
  import Ecto.Changeset

  @doc """
  Generate a new TOTP secret for a user
  """
  def generate_secret do
    :crypto.strong_rand_bytes(20) |> Base.encode32(padding: false)
  end

  @doc """
  Generate QR code data for TOTP setup
  """
  def generate_qr_code_data(user, secret) do
    config = Config.tfa_config()

    uri =
      "otpauth://totp/#{config.issuer}:#{user.email}?secret=#{secret}&issuer=#{URI.encode(config.issuer)}&digits=#{config.digits}&period=#{config.period}"

    case EQRCode.encode(uri) do
      {:ok, qr_code} ->
        svg_data = EQRCode.svg(qr_code, width: 200)
        {:ok, %{uri: uri, qr_code_svg: svg_data}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Enable two-factor authentication for a user
  """
  def enable_tfa(user, totp_code) do
    case verify_totp(user.two_factor_secret, totp_code) do
      true ->
        backup_codes = generate_backup_codes()

        changeset =
          user
          |> change(%{
            two_factor_enabled: true,
            backup_codes: backup_codes
          })

        case Repo.update(changeset) do
          {:ok, updated_user} ->
            {:ok, %{user: updated_user, backup_codes: backup_codes}}

          {:error, changeset} ->
            {:error, changeset}
        end

      false ->
        {:error, :invalid_code}
    end
  end

  @doc """
  Disable two-factor authentication for a user
  """
  def disable_tfa(user) do
    changeset =
      user
      |> change(%{
        two_factor_enabled: false,
        two_factor_secret: nil,
        backup_codes: []
      })

    Repo.update(changeset)
  end

  @doc """
  Setup two-factor authentication (generate secret but don't enable yet)
  """
  def setup_tfa(user) do
    secret = generate_secret()

    changeset =
      user
      |> change(%{two_factor_secret: secret})

    case Repo.update(changeset) do
      {:ok, updated_user} ->
        case generate_qr_code_data(updated_user, secret) do
          {:ok, qr_data} ->
            {:ok, %{user: updated_user, secret: secret, qr_data: qr_data}}

          {:error, reason} ->
            {:error, reason}
        end

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  @doc """
  Verify a TOTP code
  """
  def verify_totp(secret, code) when is_binary(secret) and is_binary(code) do
    config = Config.tfa_config()

    try do
      code_int = String.to_integer(code)
      :pot.valid_totp(code_int, secret, window: config.window)
    rescue
      ArgumentError -> false
    end
  end

  def verify_totp(_secret, _code), do: false

  @doc """
  Verify a two-factor authentication code (TOTP or backup code)
  """
  def verify_tfa_code(user, code) do
    cond do
      # Check TOTP code
      user.two_factor_secret && verify_totp(user.two_factor_secret, code) ->
        {:ok, :totp}

      # Check backup codes
      code in user.backup_codes ->
        # Remove used backup code
        remaining_codes = List.delete(user.backup_codes, code)
        changeset = change(user, %{backup_codes: remaining_codes})

        case Repo.update(changeset) do
          {:ok, _updated_user} -> {:ok, :backup_code}
          {:error, _changeset} -> {:error, :database_error}
        end

      true ->
        {:error, :invalid_code}
    end
  end

  @doc """
  Generate new backup codes for a user
  """
  def regenerate_backup_codes(user) do
    backup_codes = generate_backup_codes()

    changeset =
      user
      |> change(%{backup_codes: backup_codes})

    case Repo.update(changeset) do
      {:ok, updated_user} ->
        {:ok, %{user: updated_user, backup_codes: backup_codes}}

      {:error, changeset} ->
        {:error, changeset}
    end
  end

  @doc """
  Check if user has two-factor authentication enabled
  """
  def tfa_enabled?(user) do
    user.two_factor_enabled && user.two_factor_secret
  end

  @doc """
  Get remaining backup codes count
  """
  def backup_codes_count(user) do
    length(user.backup_codes || [])
  end

  # Private functions

  defp generate_backup_codes do
    config = Config.tfa_config()

    for _ <- 1..config.backup_codes_count do
      generate_backup_code()
    end
  end

  defp generate_backup_code do
    :crypto.strong_rand_bytes(4)
    |> Base.encode16()
    |> String.downcase()
    |> String.graphemes()
    |> Enum.chunk_every(4)
    |> Enum.map(&Enum.join/1)
    |> Enum.join("-")
  end
end
