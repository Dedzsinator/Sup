defmodule Sup.Repo.Migrations.CreateRoomMembers do
  use Ecto.Migration

  def change do
    create table(:room_members, primary_key: false) do
      add(:id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()"))
      add(:room_id, :binary_id, null: false)
      add(:user_id, :binary_id, null: false)
      add(:role, :string, null: false, default: "member")
      add(:joined_at, :utc_datetime, null: false)
      add(:permissions, :map, default: %{})

      timestamps()
    end

    create(unique_index(:room_members, [:room_id, :user_id]))
    create(index(:room_members, [:user_id]))
    create(index(:room_members, [:role]))
  end
end
