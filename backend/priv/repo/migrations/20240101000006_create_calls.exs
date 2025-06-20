defmodule Sup.Repo.Migrations.CreateCalls do
  use Ecto.Migration

  def change do
    create table(:calls, primary_key: false) do
      add(:id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()"))
      add(:caller_id, :binary_id, null: false)
      # null for direct calls
      add(:room_id, :binary_id, null: true)
      # voice, video, screen_share
      add(:type, :string, null: false)
      # connecting, ringing, active, ended, missed, declined
      add(:status, :string, null: false, default: "connecting")
      add(:started_at, :utc_datetime, null: false, default: fragment("NOW()"))
      add(:ended_at, :utc_datetime)
      # in seconds
      add(:duration, :integer)
      add(:quality_metrics, :map, default: %{})
      add(:participants, {:array, :binary_id}, default: [])
      add(:recording_url, :string)
      add(:encryption_key, :string)

      timestamps()
    end

    create(index(:calls, [:caller_id]))
    create(index(:calls, [:room_id]))
    create(index(:calls, [:status]))
    create(index(:calls, [:started_at]))
  end
end
