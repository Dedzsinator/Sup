defmodule Sup.Repo.Migrations.CreateAuditLogs do
  use Ecto.Migration

  def change do
    create table(:audit_logs, primary_key: false) do
      add(:id, :binary_id, primary_key: true)
      add(:event_type, :string, null: false)
      add(:user_id, :binary_id)
      add(:ip_address, :string)
      add(:user_agent, :text)
      add(:resource_type, :string)
      add(:resource_id, :string)
      add(:action, :string, null: false)
      add(:changes, :map)
      add(:metadata, :map)
      add(:severity, :string, null: false)
      add(:success, :boolean, null: false, default: true)
      add(:error_message, :text)
      add(:session_id, :string)
      add(:request_id, :string)

      timestamps(type: :utc_datetime)
    end

    create(index(:audit_logs, [:event_type]))
    create(index(:audit_logs, [:user_id]))
    create(index(:audit_logs, [:ip_address]))
    create(index(:audit_logs, [:action]))
    create(index(:audit_logs, [:severity]))
    create(index(:audit_logs, [:success]))
    create(index(:audit_logs, [:inserted_at]))
    create(index(:audit_logs, [:event_type, :inserted_at]))
    create(index(:audit_logs, [:user_id, :inserted_at]))
  end
end
