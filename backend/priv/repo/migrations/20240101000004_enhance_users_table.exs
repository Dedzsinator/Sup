defmodule Sup.Repo.Migrations.EnhanceUsersTable do
  use Ecto.Migration

  def change do
    alter table(:users) do
      # Profile enhancements
      # Custom display name (different from username)
      add(:display_name, :string)
      add(:bio, :text)
      add(:status_message, :string, default: "Hey there! I am using Sup.")
      add(:profile_banner_url, :string)
      # system, light, dark
      add(:theme_preference, :string, default: "system")
      add(:accent_color, :string, default: "#3B82F6")

      # Settings
      add(:notification_settings, :map,
        default: %{
          "messages" => true,
          "mentions" => true,
          "calls" => true,
          "sound" => true,
          "vibration" => true,
          "email_notifications" => false
        }
      )

      add(:privacy_settings, :map,
        default: %{
          # everyone, friends, nobody
          "online_status" => "everyone",
          "profile_visibility" => "everyone",
          "message_receipts" => true,
          "typing_indicators" => true
        }
      )

      add(:call_settings, :map,
        default: %{
          "camera_default" => true,
          "mic_default" => true,
          "noise_suppression" => true,
          "echo_cancellation" => true,
          # auto, low, medium, high
          "video_quality" => "auto"
        }
      )

      # Security and verification
      add(:email_verified, :boolean, default: false)
      add(:phone_verified, :boolean, default: false)
      add(:two_factor_enabled, :boolean, default: false)
      add(:two_factor_secret, :string)
      add(:backup_codes, {:array, :string})

      # Activity tracking
      add(:date_joined, :utc_datetime, default: fragment("NOW()"))
      # online, away, busy, invisible
      add(:activity_status, :string, default: "online")
      # For custom status like "Playing a game"
      add(:custom_activity, :map)

      # Friend system
      add(:friend_code, :string)

      # Voice/Video capabilities
      add(:voice_server_region, :string, default: "auto")
    end

    # Create indexes for performance
    create(index(:users, [:activity_status]))
    create(index(:users, [:email_verified]))
    create(unique_index(:users, [:friend_code]))
    create(index(:users, [:date_joined]))
  end
end
