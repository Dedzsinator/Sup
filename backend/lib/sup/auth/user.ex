defmodule Sup.Auth.User do
  @moduledoc """
  Enhanced user schema for authentication and comprehensive profile management.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "users" do
    # Basic auth fields
    field(:email, :string)
    field(:username, :string)
    field(:password_hash, :string)
    field(:phone, :string)

    # Profile enhancements
    field(:display_name, :string)
    field(:avatar_url, :string)
    field(:profile_banner_url, :string)
    field(:bio, :string)
    field(:status_message, :string, default: "Hey there! I am using Sup.")
    field(:theme_preference, :string, default: "system")
    field(:accent_color, :string, default: "#3B82F6")

    # Activity and presence
    field(:is_online, :boolean, default: false)
    field(:last_seen, :utc_datetime)
    field(:activity_status, :string, default: "online")
    field(:custom_activity, :map)
    field(:date_joined, :utc_datetime)

    # Settings
    field(:notification_settings, :map)
    field(:privacy_settings, :map)
    field(:call_settings, :map)

    # Security
    field(:email_verified, :boolean, default: false)
    field(:phone_verified, :boolean, default: false)
    field(:two_factor_enabled, :boolean, default: false)
    field(:two_factor_secret, :string)
    field(:backup_codes, {:array, :string})
    field(:role, :string, default: "user")
    field(:account_locked, :boolean, default: false)
    field(:locked_until, :utc_datetime)
    field(:failed_login_attempts, :integer, default: 0)
    field(:public_key, :string)
    field(:private_key_hash, :string)
    field(:session_id, :string)
    field(:refresh_token_hash, :string)

    # Communication
    field(:push_token, :string)
    field(:friend_code, :string)
    field(:voice_server_region, :string, default: "auto")

    timestamps()
  end

  def changeset(user, attrs) do
    user
    |> cast(attrs, [
      :email,
      :username,
      :display_name,
      :bio,
      :status_message,
      :avatar_url,
      :profile_banner_url,
      :theme_preference,
      :accent_color,
      :activity_status,
      :custom_activity,
      :notification_settings,
      :privacy_settings,
      :call_settings,
      :push_token,
      :is_online,
      :last_seen,
      :role,
      :account_locked,
      :locked_until,
      :failed_login_attempts,
      :public_key,
      :session_id
    ])
    |> validate_format(:email, ~r/^[^\s]+@[^\s]+\.[^\s]+$/)
    |> validate_length(:username, min: 2, max: 32)
    |> validate_length(:display_name, min: 1, max: 64)
    |> validate_length(:bio, max: 500)
    |> validate_length(:status_message, max: 128)
    |> validate_inclusion(:theme_preference, ["system", "light", "dark"])
    |> validate_inclusion(:activity_status, ["online", "away", "busy", "invisible"])
    |> validate_inclusion(:role, ["admin", "moderator", "user", "guest"])
    |> unique_constraint(:email)
    |> unique_constraint(:username)
    |> unique_constraint(:friend_code)
  end

  def registration_changeset(user, attrs) do
    user
    |> cast(attrs, [:email, :username, :password_hash, :display_name, :avatar_url])
    |> validate_required([:email, :username, :password_hash])
    |> validate_format(:email, ~r/^[^\s]+@[^\s]+\.[^\s]+$/)
    |> validate_length(:username, min: 2, max: 32)
    |> validate_length(:display_name, min: 1, max: 64)
    |> validate_length(:password_hash, min: 8)
    |> unique_constraint(:email)
    |> unique_constraint(:username)
    |> unique_constraint(:friend_code)
    |> put_default_settings()
    |> generate_friend_code()
  end

  def profile_changeset(user, attrs) do
    user
    |> cast(attrs, [
      :display_name,
      :bio,
      :status_message,
      :avatar_url,
      :profile_banner_url,
      :theme_preference,
      :accent_color,
      :activity_status,
      :custom_activity
    ])
    |> validate_length(:display_name, min: 1, max: 64)
    |> validate_length(:bio, max: 500)
    |> validate_length(:status_message, max: 128)
    |> validate_inclusion(:theme_preference, ["system", "light", "dark"])
    |> validate_inclusion(:activity_status, ["online", "away", "busy", "invisible"])
  end

  def settings_changeset(user, attrs) do
    user
    |> cast(attrs, [:notification_settings, :privacy_settings, :call_settings])
    |> validate_settings()
  end

  def security_changeset(user, attrs) do
    user
    |> cast(attrs, [
      :email_verified,
      :phone_verified,
      :two_factor_enabled,
      :two_factor_secret,
      :backup_codes,
      :role,
      :account_locked,
      :locked_until,
      :failed_login_attempts,
      :public_key,
      :private_key_hash,
      :session_id,
      :refresh_token_hash
    ])
  end

  def public_fields(user) do
    %{
      id: user.id,
      email: user.email,
      username: user.username,
      display_name: user.display_name || user.username,
      avatar_url: user.avatar_url,
      profile_banner_url: user.profile_banner_url,
      bio: user.bio,
      status_message: user.status_message,
      is_online: user.is_online,
      last_seen: user.last_seen,
      activity_status: user.activity_status,
      custom_activity: user.custom_activity,
      theme_preference: user.theme_preference,
      accent_color: user.accent_color,
      date_joined: user.date_joined,
      friend_code: user.friend_code,
      email_verified: user.email_verified,
      phone_verified: user.phone_verified,
      two_factor_enabled: user.two_factor_enabled
    }
  end

  def friends_fields(user) do
    %{
      id: user.id,
      username: user.username,
      display_name: user.display_name || user.username,
      avatar_url: user.avatar_url,
      status_message: user.status_message,
      is_online: user.is_online,
      last_seen: user.last_seen,
      activity_status: user.activity_status,
      custom_activity: user.custom_activity
    }
  end

  defp put_default_settings(changeset) do
    changeset
    |> put_change(:notification_settings, %{
      "messages" => true,
      "mentions" => true,
      "calls" => true,
      "sound" => true,
      "vibration" => true,
      "email_notifications" => false
    })
    |> put_change(:privacy_settings, %{
      "online_status" => "everyone",
      "profile_visibility" => "everyone",
      "message_receipts" => true,
      "typing_indicators" => true
    })
    |> put_change(:call_settings, %{
      "camera_default" => true,
      "mic_default" => true,
      "noise_suppression" => true,
      "echo_cancellation" => true,
      "video_quality" => "auto"
    })
    |> put_change(:date_joined, DateTime.utc_now())
  end

  defp generate_friend_code(changeset) do
    # Generate a unique 8-character friend code
    friend_code =
      :crypto.strong_rand_bytes(4)
      |> Base.encode16()
      |> String.downcase()

    put_change(changeset, :friend_code, friend_code)
  end

  defp validate_settings(changeset) do
    # Add validation for settings structure if needed
    changeset
  end
end
