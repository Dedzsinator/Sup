defmodule Sup.Repo.Migrations.CreateRooms do
  use Ecto.Migration

  def change do
    create table(:rooms, primary_key: false) do
      add(:id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()"))
      add(:name, :string, null: false)
      add(:description, :text)
      add(:type, :string, null: false)
      add(:is_private, :boolean, default: false)
      add(:created_by, :binary_id, null: false)
      add(:avatar_url, :string)
      add(:settings, :map, default: %{})

      timestamps()
    end

    create(index(:rooms, [:type]))
    create(index(:rooms, [:created_by]))
    create(index(:rooms, [:is_private]))
  end
end
